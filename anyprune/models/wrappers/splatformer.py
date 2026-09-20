"""
Trainable PyTorch module to run SplatFormer over a set of Gaussians, 
refining them in place, with a simplified interface.

Two optional additions to the released architecture, which reads a
Gaussian as 23 channels (means, log scales, opacity logit, quaternion,
SH coefficients):

  - an importance input: the RadSplat score of every Gaussian, the
    peak blending weight it carried over the context views, as a 24th
    channel, read as log(max(S, eps)) - log(reference) so that it is
    zero at the reference cut, negative below and positive above. The
    extra column of the embedding and of the first layer of every
    output head starts at zero, so a widened network starts exactly
    where the checkpoint it was built from left it.
  - a mask head: one more output head, on the same (backbone + input)
    features the six existing heads read, predicting a per-Gaussian
    offset to that same log score. Its logit, log score plus offset, is
    what a Gumbel-Sigmoid turns into a keep mask (see
    anyprune.training.learned_pruning); it starts at zero, so the mask
    starts as a smooth version of the reference threshold.

To use the module for inference, it must be used in single precision 
mode.
"""
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
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

# The name of the importance channel among SplatFormer's input features
# and of the mask head among its output heads, which is also the key
# the head's learning rate is looked up under.
IMPORTANCE_FEATURE = "importance"
MASK_HEAD = "mask"
# As wide as SplatFormer's own heads
_MASK_HEAD_WIDTH = 128


def _widen_linear(linear: nn.Linear, extra: int) -> nn.Linear:
    """
    The same linear layer reading 'extra' more input channels, whose
    weights start at zero: on the inputs it already read it computes
    exactly what it did.
    """
    widened = nn.Linear(
        linear.in_features + extra, linear.out_features, bias=linear.bias is not None
    )
    with torch.no_grad():
        widened.weight.zero_()
        widened.weight[:, :linear.in_features] = linear.weight
        if linear.bias is not None:
            widened.bias.copy_(linear.bias)
    return widened.to(linear.weight.device)


def _mask_head(input_dim: int, width: int = _MASK_HEAD_WIDTH) -> nn.Sequential:
    """
    The mask head, shaped like SplatFormer's own heads so that whatever
    handles those handles it (zeroing, state dicts, learning rates),
    with its last layer at zero.
    """
    head = nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, 1),
    )
    head[-1].weight.data.zero_()
    head[-1].bias.data.zero_()
    return head


class _HeadInputCapture:
    """
    Hands over the (N, D) tensor SplatFormer feeds to its output heads,
    by way of a forward pre-hook on one of them, attached only around
    the forward whose features are wanted and shape-checked when taken:
    a tensor left over from another field would otherwise be used
    silently, and would pin that field's autograd graph.
    """

    def __init__(self, module: nn.Module, input_dim: int):
        self.module = module
        self.input_dim = input_dim
        self.captured: Optional[Tensor] = None

    def _hook(self, module, args):
        features = args[0]
        assert features.dim() == 2 and features.shape[1] == self.input_dim, (
            f"Expected the head to be fed (N, {self.input_dim}), got "
            f"{tuple(features.shape)}"
        )
        self.captured = features

    @contextmanager
    def attached(self) -> Iterator["_HeadInputCapture"]:
        handle = self.module.register_forward_pre_hook(self._hook)
        try:
            yield self
        finally:
            handle.remove()
            self.captured = None

    def take(self, num_gaussians: int) -> Tensor:
        features, self.captured = self.captured, None
        assert features is not None, "The output head did not run"
        assert features.shape[0] == num_gaussians, (
            f"Captured {features.shape[0]:,} rows for {num_gaussians:,} Gaussians"
        )
        return features


class SplatFormer(nn.Module):
    def __init__(
        self,
        pretrained_ckpt: Optional[str],
        quiet: bool,
        config_file: Path = SPLATFORMER_MODEL_CONFIG,
        zero_output_heads: bool = False,
        gradient_checkpointing: bool = False,
        batch_statistics: bool = False,
        importance_input: bool = False,
        mask_head: bool = False,
        importance_reference: float = 0.01,
        importance_eps: float = 1e-6,
    ):
        """
        'importance_input' and 'mask_head' are the two additions
        described at the top of the module; both read the score handed
        to refine() as log(max(S, importance_eps)) - log(importance_reference).
        A checkpoint written before either addition existed loads into
        either, with the new weights at their initialization; one
        written with them loads exactly.
        """
        super().__init__()
        self.quiet = quiet
        self.batch_statistics = batch_statistics
        self.importance_input = importance_input
        self.mask_head = mask_head
        self.importance_reference = importance_reference
        self.importance_eps = importance_eps
        assert importance_reference > 0.0 and importance_eps > 0.0, (
            f"The score is read in logs, so its reference and floor have to "
            f"be positive: got {importance_reference} and {importance_eps}"
        )
        self._capture = None
        with _muted(quiet):
            self.model = build_splatformer(config_file)
            if importance_input:
                self._add_importance_input()
            if mask_head:
                self.model.features_outputhead[MASK_HEAD] = _mask_head(self._head_input_dim)
                self._capture = _HeadInputCapture(
                    self.model.features_outputhead["opacities"], self._head_input_dim
                )
            if pretrained_ckpt is not None:
                self._load_weights(pretrained_ckpt)
            if zero_output_heads:
                self._zero_output_heads()
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self._checkpoint_backbone_blocks()
        if batch_statistics:
            for module in self._batch_norms():
                module.momentum = 0.0

    def _batch_norms(self):
        return [m for m in self.modules() if isinstance(m, _BatchNorm)]

    def train(self, mode: bool = True):
        """
        Switch modes as usual, except that with 'batch_statistics' the
        BatchNorm layers stay on the statistics of the field they are
        given even in eval mode, instead of the running averages of
        training, which a held-out field of another size or view count
        lands far from. Their running averages are pinned (momentum 0)
        so that no field moves them.
        """
        super().train(mode)
        if not mode and self.batch_statistics:
            for module in self._batch_norms():
                module.train()
        return self

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

    @property
    def _head_input_dim(self) -> int:
        """What every output head reads: the backbone's features and the raw input."""
        return self.model.backbone.output_dim + self.model.gs_features_dim

    @property
    def reads_score(self) -> bool:
        """Whether refine() has to be handed a score."""
        return self.importance_input or self.mask_head

    def _widened_layers(self) -> Dict[str, nn.Linear]:
        """
        The layers the importance channel widens, by the prefix of their
        state-dict keys: the backbone's embedding and the first layer of
        every head.
        """
        layers = {"backbone.backbone.embedding.0": self.model.backbone.backbone.embedding[0]}
        for name, head in self.model.features_outputhead.items():
            layers[f"features_outputhead.{name}.0"] = head[0]
        return layers

    def _add_importance_input(self):
        """
        Register the importance channel as the last input feature and
        widen everything that reads the raw input by one column.
        """
        model = self.model
        assert IMPORTANCE_FEATURE not in model.input_features
        # The channel table is a module-level dictionary of SplatFormer's
        # feature predictor, reached through the class's own namespace:
        # its module only exists inside its own root and is not left in
        # sys.modules
        type(model).forward.__globals__["FEATURE2CHANNEL"][IMPORTANCE_FEATURE] = 1
        model.input_features = list(model.input_features) + [IMPORTANCE_FEATURE]
        model.gs_features_dim += 1
        embedding = model.backbone.backbone.embedding
        assert isinstance(embedding[0], nn.Linear), "Expected the MLP embedding of ptv3.gin"
        embedding.add_module("0", _widen_linear(embedding[0], 1))
        for head in model.features_outputhead.values():
            head[0] = _widen_linear(head[0], 1)

    def _fit_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        A checkpoint as this architecture can load it: one written
        without the importance channel gets a zero column where the
        channel is read, and one written without the mask head gets the
        head as it is initialized here. Anything else is left to the
        strict load to complain about.
        """
        state_dict = dict(state_dict)
        if self.importance_input:
            for prefix, layer in self._widened_layers().items():
                weight = state_dict.get(f"{prefix}.weight")
                if weight is not None and weight.shape[1] == layer.in_features - 1:
                    state_dict[f"{prefix}.weight"] = torch.cat(
                        [weight, weight.new_zeros(weight.shape[0], 1)], dim=1
                    )
        if self.mask_head:
            prefix = f"features_outputhead.{MASK_HEAD}."
            if not any(key.startswith(prefix) for key in state_dict):
                state_dict.update({
                    key: value for key, value in self.model.state_dict().items()
                    if key.startswith(prefix)
                })
        return state_dict

    def _load_weights(self, pretrained_ckpt: str):
        """
        Fill the model with the weights of a checkpoint, named as a path
        on this machine.
        """
        state_dict = torch.load(pretrained_ckpt, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        state_dict = self._fit_state_dict(state_dict)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        assert not missing and not unexpected, (
            f"{pretrained_ckpt} does not fit this architecture: "
            f"{len(missing)} weights missing (e.g. {missing[:3]}) and "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}). "
            f"Check that it was trained with the gin config being used, "
            f"and with the same importance input and mask head."
        )

    def importance_feature(self, score: Tensor) -> Tensor:
        """
        The (N, 1) channel a score is read as: its log, floored at
        importance_eps, centred on importance_reference.
        """
        return (
            score.float().clamp_min(self.importance_eps).log()
            - math.log(self.importance_reference)
        ).unsqueeze(-1)

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

    def refine(
        self, gaussians: Gaussians, score: Optional[Tensor] = None
    ) -> Tuple[Gaussians, Optional[Tensor]]:
        """
        Takes a set of Gaussians and returns the refined set, in the
        same frame and with the same cameras still valid, so that the
        two can be rasterized against each other, alongside the mask
        head's logit for every Gaussian when there is a head and None
        when there is not.

        'score' is the (N,) RadSplat score of every Gaussian, needed
        whenever the network reads one (see reads_score) and ignored
        otherwise.

        The normalization into the unit cube that SplatFormer's voxel
        grid needs happens inside and is undone on the way out, which is
        what lets the caller keep working in the frame the Gaussians
        arrived in.
        """
        gs, scaler = self._to_parameters(gaussians)
        gs, unseen_harmonics = self._to_model_sh_degree(gs)
        importance = None
        if self.reads_score:
            assert score is not None, (
                "This SplatFormer reads the importance score, refine() has to be handed one"
            )
            assert score.shape == (gaussians.num_gaussians,), (
                f"One score per Gaussian: got {tuple(score.shape)} for {gaussians.num_gaussians:,}"
            )
            importance = self.importance_feature(score)
            if self.importance_input:
                gs[IMPORTANCE_FEATURE] = importance.to(gs["opacities"].dtype)

        features = None
        with self._capture.attached() if self.mask_head else nullcontext():
            with _muted(self.quiet):
                # The model works on batches of scenes, we do one at a time
                refined = self.model(batch_normalized_gs=[gs], batch_scene_idx=[0])[0]
            if self.mask_head:
                features = self._capture.take(gaussians.num_gaussians)

        # Put back together in single precision, the head included:
        # its logits go through a Gumbel-Sigmoid that saturates in half
        with torch.cuda.amp.autocast(enabled=False):
            refined = {key: value.float() for key, value in refined.items()}
            refined["features_rest"] = torch.cat(
                [refined["features_rest"], unseen_harmonics], dim=1
            )
            refined = self._from_parameters(refined, scaler)
            logits = None
            if self.mask_head:
                offset = self.model.features_outputhead[MASK_HEAD](features.float())
                logits = importance.squeeze(-1) + offset.squeeze(-1)
        return refined, logits

    def forward(self, gaussians: Gaussians, score: Optional[Tensor] = None) -> Gaussians:
        """
        The refined set alone, for a network without a mask head: with
        one, what survives refinement is the mask's call and the caller
        has to read it off refine().
        """
        assert not self.mask_head, (
            "A SplatFormer with a mask head refines through refine(), which also "
            "returns the mask"
        )
        return self.refine(gaussians, score)[0]


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
