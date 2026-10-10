"""
A PyTorch module to run ReSplat to produce Gaussians from a set of
posed views, all with a simplified interface.
"""
import math
from types import SimpleNamespace
from typing import Optional

import gsplat
import torch
import torch.nn as nn
import torch.nn.functional as F
from dacite import from_dict
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import Tensor

from anyprune.utils import _muted
with _muted(True): from ..utils import (
    EncoderReSplatCfg, RESPLAT_ROOT, build_resplat_encoder,
)
from ...gaussians import Gaussians


# Both sides have to be a whole number of the encoder's patches, which
# its data shim would otherwise crop them to (shim_patch_size *
# downscale_factor in its DL3DV configuration)
PATCH_SIZE = 64
# How many times the recurrent module refines the initial Gaussians,
# which is what ReSplat's own inference does with the 8-view DL3DV models
NUM_REFINE = 4
# The depth range ReSplat's DL3DV configuration hands the encoder, in
# the units of the dataset's poses: it sweeps its depth candidates in it
NEAR, FAR = 0.01, 200.0

# ReSplat is configured through Hydra, so rather than repeat its
# settings we read the encoder file it ships and apply what its DL3DV
# experiment overrides on top of it.
_ENCODER_CONFIG = RESPLAT_ROOT / "config" / "model" / "encoder" / "resplat.yaml"
_EXPERIMENT_CONFIG = RESPLAT_ROOT / "config" / "experiment" / "dl3dv.yaml"

# OpenCV <-> OpenGL camera axes, the same matrix both ways
_FLIP_YZ = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def _encoder_config(num_refine: int) -> EncoderReSplatCfg:
    """
    The configuration of ReSplat's encoder for its DL3DV checkpoints,
    typed the way ReSplat's own entry point types it.
    """
    cfg = OmegaConf.load(_ENCODER_CONFIG)
    experiment = OmegaConf.load(_EXPERIMENT_CONFIG).model.encoder
    cfg = OmegaConf.merge(cfg, experiment, {"num_refine": num_refine})
    return from_dict(EncoderReSplatCfg, OmegaConf.to_container(cfg))


class _Renderer:
    """
    What ReSplat's refinement renders its Gaussians with to measure its
    error on the context views: its gsplat decoder, with the same
    rasterizer settings, for the single scene and the gsplat version of
    this project, which takes no batch dimension.
    """

    @staticmethod
    def forward(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode=None):
        assert extrinsics.shape[0] == 1, "ReSplat is only ever run on one scene at a time here"
        height, width = image_shape
        Ks = intrinsics[0].clone()
        Ks[:, 0] *= width
        Ks[:, 1] *= height
        colors = gaussians.harmonics[0].transpose(-2, -1)
        colors, _, _ = gsplat.rasterization(
            means=gaussians.means[0],
            quats=None,
            scales=None,
            opacities=gaussians.opacities[0],
            colors=colors,
            sh_degree=int(math.sqrt(colors.shape[-2])) - 1,
            viewmats=extrinsics[0].inverse(),
            Ks=Ks,
            width=width,
            height=height,
            near_plane=near[0, 0].item(),
            far_plane=far[0, 0].item(),
            eps2d=0.1,
            rasterize_mode="antialiased",
            packed=True,
            covars=gaussians.covariances[0],
        )
        return SimpleNamespace(color=colors.permute(0, 3, 1, 2).unsqueeze(0))


class FrozenReSplat(nn.Module):
    # Unlike the pose-free reconstructors, ReSplat is handed the cameras
    # of its context views rather than predicting them
    needs_cameras = True

    def __init__(
        self,
        pretrained_ckpt: str,
        quiet: bool,
        num_refine: int = NUM_REFINE,
        short_side: Optional[int] = None,
    ):
        """
        'short_side' is the short side, in pixels, the frames are
        resized to before the encoder sees them, or None to show it the
        frames at their own resolution.

        The DL3DV checkpoints are trained on 256x448 frames, but on the
        448x448 frames of this project it does better at their own
        resolution than with the short side brought to 256: by about
        2 dB of PSNR on held-out views (4 test scenes, 2 and 4 views),
        and with 3x as many Gaussians.
        """
        super().__init__()
        self.quiet = quiet
        self.short_side = short_side
        with _muted(quiet):
            self.model = build_resplat_encoder(_encoder_config(num_refine))
            self._load_weights(pretrained_ckpt)
        self.model.eval()
        self.model.requires_grad_(False)

    def _load_weights(self, pretrained_ckpt: str):
        """
        Fill the encoder with the weights of a checkpoint, named as the
        file of a Hugging Face repository it lives in, such as
        'haofeixu/resplat/resplat-base-dl3dv-256x448-view8-1934a04c.pth'.
        It is downloaded on first use and read from the cache afterwards.

        The checkpoints hold the state of ReSplat's Lightning wrapper,
        whose only weights are the encoder's.
        """
        repo_id, filename = pretrained_ckpt.rsplit("/", 1)
        checkpoint = torch.load(
            hf_hub_download(repo_id=repo_id, filename=filename), map_location="cpu"
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        prefix = "encoder."
        state_dict = {
            key[len(prefix):]: weights for key, weights in state_dict.items()
            if key.startswith(prefix)
        }
        assert state_dict, f"No encoder weights in {pretrained_ckpt}"
        self.model.load_state_dict(state_dict)

    def _native_shape(self, image_shape: tuple[int, int]) -> tuple[int, int]:
        """
        The shape the encoder is shown frames of 'image_shape' at: the
        short side brought to 'short_side' if there is one, keeping the
        aspect ratio, and both sides a whole number of patches.
        """
        if self.short_side is None:
            scale = 1.0
        else:
            scale = self.short_side / min(image_shape)
        return tuple(
            max(round(side * scale / PATCH_SIZE), 1) * PATCH_SIZE
            for side in image_shape
        )

    @staticmethod
    def _normalized_intrinsics(
        intrinsics: Tensor, image_shape: tuple[int, int], native_shape: tuple[int, int]
    ) -> Tensor:
        """
        Pinhole matrices in pixels of frames of 'image_shape' rewritten
        the way ReSplat takes them: for those frames resized to
        'native_shape', with the first row divided by the width and the
        second by the height.
        """
        normalized = intrinsics.clone()
        for row, (side, native_side) in enumerate(zip(image_shape, native_shape)):
            # Rows are x then y, sides height then width
            row = 1 - row
            scale = native_side / side
            # Pixel centers sit at integer coordinates plus a half
            normalized[:, row, row] *= scale
            normalized[:, row, 2] = (normalized[:, row, 2] + 0.5) * scale - 0.5
            normalized[:, row] /= native_side
        return normalized

    def forward(self, context_images: Tensor, poses: Tensor, intrinsics: Tensor):
        """
        Takes a (V, 3, H, W) tensor of images in [0, 1], with their
        (V, 4, 4) camera-to-world poses and their (V, 3, 3) pinhole
        intrinsics in pixels of those images, both in the convention of
        Gaussians.rasterize(), and returns a tuple
        (poses, intrinsics, gaussians).

        ReSplat reconstructs in the frame of the middle context view, as
        it was trained to, and its harmonics are only right in that
        frame: the poses that come back are the ones passed in, written
        in it, and the intrinsics are the ones passed in.
        """
        assert context_images.shape[1] == 3, f"Expected (V, 3, H, W) shape for context images, got {context_images.shape}"
        assert context_images.dim() == 4, f"context_images should have dim 4, got {context_images.dim()}"
        num_views = context_images.shape[0]
        poses = poses.reshape(num_views, 4, 4).to(context_images)
        intrinsics = intrinsics.reshape(num_views, 3, 3).to(context_images)

        image_shape = tuple(context_images.shape[-2:])
        native_shape = self._native_shape(image_shape)
        views = context_images
        if native_shape != image_shape:
            views = F.interpolate(
                views, size=native_shape, mode="bilinear", align_corners=False, antialias=True,
            )

        # OpenGL -> OpenCV camera axes, then into the frame of the
        # middle context view, which is where ReSplat's evaluation on
        # DL3DV puts its scenes (pose_align_middle_view)
        # (in double precision: the context views of a dense capture are
        # centimeters apart, a long way from the origin)
        flip = _FLIP_YZ.to(poses.device, torch.float64)
        extrinsics = poses.double() @ flip
        extrinsics = torch.linalg.inv(extrinsics[num_views // 2]) @ extrinsics
        extrinsics = extrinsics.to(poses.dtype)

        context = {
            "image": views.unsqueeze(0),
            "extrinsics": extrinsics.unsqueeze(0),
            "intrinsics": self._normalized_intrinsics(intrinsics, image_shape, native_shape).unsqueeze(0),
            "near": torch.full((1, num_views), NEAR).to(views),
            "far": torch.full((1, num_views), FAR).to(views),
        }
        with _muted(self.quiet):
            output = self.model(context, global_step=0, deterministic=False)
            gaussians = output["gaussians"]
            if self.model.cfg.num_refine > 0:
                # Refined on its error at rendering the context views,
                # the only ones it is handed
                gaussians = self.model.forward_update(
                    context, None, output["condition_features"], gaussians, _Renderer,
                )["gaussian"][-1]

        return extrinsics @ flip.to(extrinsics), intrinsics, Gaussians.from_resplat(gaussians)
