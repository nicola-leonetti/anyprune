"""
Trainable PyTorch module to run SplatFormer over a set of Gaussians, 
refining them in place, with a simplified interface.

To use the module for inference, it must be used in single precision 
mode.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from anyprune.utils import _muted
with _muted(True): from ..utils import (
    SPLATFORMER_MODEL_CONFIG, MinMaxScaler, build_anysplat_covariance,
    build_splatformer,
)
from ...gaussians import Gaussians


# For numerical stability, this value is used instead of 0 for scale
# and opacity when its value is mathematically 0.
_MIN_SCALE = 1e-12
_MIN_OPACITY = 1e-6


class SplatFormer(nn.Module):
    def __init__(
        self,
        pretrained_ckpt: Optional[str],
        quiet: bool,
        config_file: Path = SPLATFORMER_MODEL_CONFIG,
        zero_output_heads: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.quiet = quiet
        with _muted(quiet):
            self.model = build_splatformer(config_file)
            if pretrained_ckpt is not None:
                self._load_weights(pretrained_ckpt)
            if zero_output_heads:
                self._zero_output_heads()
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self._checkpoint_backbone_blocks()

    def _checkpoint_backbone_blocks(self):
        """
        Recompute each transformer block of the backbone in the backward
        pass instead of holding its activations from the forward one.

        This is done to save memory and fit more gaussians on the same 
        GPU.
        """
        with _muted(True):
            from pointcept.models.point_transformer_v3 import Block
        for block in self.model.backbone.modules():
            if isinstance(block, Block):
                block.forward = _checkpointed_block(block)

    def _zero_output_heads(self):
        """
        Zero the last layer of every output head. This is used to make
        the network start from a state in which it predicts no 
        adjustment for every gaussian parameter in the input.
        """
        for head in self.model.features_outputhead.values():
            head[-1].weight.data.zero_()
            head[-1].bias.data.zero_()

    def _load_weights(self, pretrained_ckpt: str):
        """
        Fill the model with the weights of a checkpoint, named as a path
        on this machine.
        """
        state_dict = torch.load(pretrained_ckpt, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        assert not missing and not unexpected, (
            f"{pretrained_ckpt} does not fit this architecture: "
            f"{len(missing)} weights missing (e.g. {missing[:3]}) and "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}). "
            f"Check that it was trained with the gin config being used."
        )

    @property
    def sh_degree(self) -> int:
        """
        The spherical harmonic degree the network reads and writes,
        which is fixed by the weights rather than by its input.
        """
        return self.model.sh_degree

    @staticmethod
    def _to_parameters(gaussians: Gaussians) -> Tuple[Dict[str, Tensor], MinMaxScaler]:
        """
        Write a set of Gaussians the way SplatFormer reads them, as the
        raw parameters of a splatfacto model normalized into the unit
        cube, and return them alongside the scaler that undoes the
        normalization.
        """
        scaler = MinMaxScaler()
        # Rounding can leave a mean a hair outside the box the scaler
        # fitted, which would land it outside the voxel grid
        means = scaler.fit_transform(gaussians.means).clamp(0.0, 1.0)
        scales = gaussians.scales.clamp_min(_MIN_SCALE).log() + torch.log(scaler.scale_)
        opacities = gaussians.opacities.clamp(_MIN_OPACITY, 1.0 - _MIN_OPACITY)
        return {
            "means": means,
            "scales": scales,
            "opacities": opacities.logit().unsqueeze(-1),
            "quats": gaussians.rotations[..., [3, 0, 1, 2]],
            "features_dc": gaussians.harmonics[..., 0],
            # Made contiguous because SplatFormer flattens it with a view
            "features_rest": gaussians.harmonics[..., 1:].transpose(-2, -1).contiguous(),
        }, scaler

    @staticmethod
    def _from_parameters(
        gs: Dict[str, Tensor], scaler: MinMaxScaler
    ) -> Gaussians:
        """
        Read back what _to_parameters() writes, undoing both the
        normalization and the reparametrizations, so that the Gaussians 
        come out in the frame their cameras are still written in.
        The covariances are rebuilt.
        """
        scales = torch.exp(gs["scales"] - torch.log(scaler.scale_))
        rotations = gs["quats"] / gs["quats"].norm(dim=-1, keepdim=True)
        # Back from nerfstudio's real-part-first quaternions to the
        # real-part-last ones build_anysplat_covariance() expects
        rotations = rotations[..., [1, 2, 3, 0]]

        features_dc = gs["features_dc"].unsqueeze(-1)          # (N, 3, 1)
        features_rest = gs["features_rest"].transpose(-2, -1)  # (N, 3, d_sh - 1)
        harmonics = torch.cat([features_dc, features_rest], dim=-1)

        return Gaussians(
            means=scaler.inverse_transform(gs["means"]),
            covariances=build_anysplat_covariance(scales, rotations),
            harmonics=harmonics,
            opacities=torch.sigmoid(gs["opacities"]).squeeze(-1),
            scales=scales,
            rotations=rotations,
        )

    def _to_model_sh_degree(
        self, gs: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Trim or pad the harmonics of a SplatFormer parameter dictionary
        to the degree the network was trained at, returning it alongside
        the coefficients the network will not see.
        """
        wanted = (self.sh_degree + 1) ** 2 - 1
        harmonics = gs["features_rest"] # (N, d_sh - 1, 3)
        seen, unseen = harmonics[:, :wanted], harmonics[:, wanted:]
        if seen.shape[1] < wanted:
            padding = seen.new_zeros(seen.shape[0], wanted - seen.shape[1], 3)
            seen = torch.cat([seen, padding], dim=1)
        return {**gs, "features_rest": seen.contiguous()}, unseen

    def forward(self, gaussians: Gaussians) -> Gaussians:
        """
        Takes a set of Gaussians and returns the refined set, in the
        same frame and with the same cameras still valid, so that the
        two can be rasterized against each other.

        The normalization into the unit cube that SplatFormer's voxel
        grid needs happens inside and is undone on the way out, which is
        what lets the caller keep working in the frame the Gaussians
        arrived in.
        """
        gs, scaler = self._to_parameters(gaussians)
        gs, unseen_harmonics = self._to_model_sh_degree(gs)

        with _muted(self.quiet):
            # The model works on batches of scenes, we do one at a time
            refined = self.model(batch_normalized_gs=[gs], batch_scene_idx=[0])[0]

        # Put back together in single precision
        with torch.cuda.amp.autocast(enabled=False):
            refined = {key: value.float() for key, value in refined.items()}
            refined["features_rest"] = torch.cat(
                [refined["features_rest"], unseen_harmonics], dim=1
            )
            return self._from_parameters(refined, scaler)


def _checkpointed_block(block: nn.Module):
    """
    A block's forward that recomputes itself in the backward pass.

    Falls back to the block as it is whenever there is nothing to
    recompute for (e.g. in eval mode, under no_grad, or on a Point whose
    features are not part of a graph).
    """
    inner = block.forward

    def forward(point):
        if not (block.training and torch.is_grad_enabled() and point.feat.requires_grad):
            return inner(point)

        def run(feat, sparse_feat):
            local = point.__class__(dict(point))
            local.feat = feat
            local.sparse_conv_feat = point.sparse_conv_feat.replace_feature(sparse_feat)
            return inner(local).feat

        feat = checkpoint(
            run, point.feat, point.sparse_conv_feat.features, use_reentrant=False
        )
        refined = point.__class__(dict(point))
        refined.feat = feat
        refined.sparse_conv_feat = point.sparse_conv_feat.replace_feature(feat)
        return refined

    return forward
