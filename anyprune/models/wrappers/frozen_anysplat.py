"""
A PyTorch module to run AnySplat to produce Gaussians from a set of 
views, all with a simplified interface.
"""
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from anyprune.utils import _muted
with _muted(True): from ..utils import build_anysplat
from ...gaussians import Gaussians


class FrozenAnySplat(nn.Module):
    def __init__(
        self, pretrained_ckpt: str, quiet: bool, voxelize: bool = False
    ):
        """
        Build AnySplat from a checkpoint, deciding once and for all
        whether it fuses its Gaussians onto a voxel grid.
        """
        super().__init__()
        self.quiet = quiet
        self.voxelize = voxelize
        with _muted(quiet):
            self.model = build_anysplat(pretrained_ckpt)
        self.model.eval()
        self.model.requires_grad_(False)

    @contextmanager
    def _voxelization(self, enabled: bool):
        """
        Context manager to temporarily override the encoder's 
        voxelization flag while avoiding making a second copy of the 
        weights.
        """
        assert enabled in (True, False), (
            f"voxelize has to be True or False, got {enabled!r}"
        )
        cfg = self.model.encoder.cfg
        if enabled == cfg.voxelize:
            yield
            return
        previous = cfg.voxelize
        cfg.voxelize = enabled
        try:
            yield
        finally:
            cfg.voxelize = previous

    @staticmethod
    @contextmanager
    def _cache_kept():
        """
        Disable torch.cuda.empty_cache() while the encoder is running.
        This is an optimization because we expect CUDA expandable 
        segments to be enabled.
        """
        original = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None
        try: yield
        finally: torch.cuda.empty_cache = original

    @staticmethod
    def _to_dl3dv_convention(
        pred_pose: dict, image_shape: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        """
        Convert AnySplat's predicted cameras to the DL3DV convention the
        rest of the project uses, so that they can go straight into
        Gaussians.rasterize().

        AnySplat predicts an 'extrinsic' (B, V, 4, 4) of camera-to-world
        matrices with OpenCV axes (+Y down, +Z forwards) and an
        'intrinsic' (B, V, 3, 3) whose first two rows are divided by the
        image width and height. DL3DV instead uses OpenGL camera axes
        (+Y up, +Z backwards) and intrinsics in pixels, so we flip the Y
        and Z axes and scale the intrinsics back up.
        """
        height, width = image_shape

        poses = pred_pose["extrinsic"][0]
        opencv_to_opengl = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]).to(poses))
        poses = poses @ opencv_to_opengl

        intrinsics = pred_pose["intrinsic"][0].clone()
        intrinsics[:, 0] *= width
        intrinsics[:, 1] *= height

        return poses, intrinsics

    def forward(
        self, context_images: Tensor, voxelize: Optional[bool] = None
    ):
        """
        Takes a (V, 3, H, W) tensor of images in [0, 1] and returns a
        tuple (poses, intrinsics, gaussians).

        'voxelize' can be overwritten here, or it can be left to None to
        make 'forward' default to the value the module was built with.

        Poses is a (V, 4, 4) tensor of camera-to-world matrices and
        intrinsics a (V, 3, 3) tensor of pinhole matrices, both already
        in the convention that Gaussians.rasterize() expects.
        """
        assert context_images.shape[1] == 3, f"Expected (V, 3, H, W) shape for context images, got {context_images.shape}"
        assert context_images.dim() == 4, f"context_images should have dim 4, got {context_images.dim()}"
        voxelize = self.voxelize if voxelize is None else voxelize
        with _muted(self.quiet), self._voxelization(voxelize), self._cache_kept():
            # The encoder works on batches of scenes, we do one at a time
            encoder_output = self.model.encoder(
                context_images.unsqueeze(0), global_step=0, visualization_dump=None
            )
        poses, intrinsics = self._to_dl3dv_convention(
            encoder_output.pred_context_pose, context_images.shape[-2:]
        )
        gaussians = Gaussians.from_anysplat(encoder_output.gaussians)
        return poses, intrinsics, gaussians
