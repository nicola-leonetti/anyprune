"""
A PyTorch module to run AnySplat to produce Gaussians from a set of 
views, all with a simplified interface.
"""
import torch
import torch.nn as nn
from torch import Tensor

from anyprune.utils import _muted
with _muted(True): from ..models import AnySplat
from ..gaussians import Gaussians


class FrozenAnySplat(nn.Module):
    def __init__(self, pretrained_ckpt: str, quiet: bool):
        super().__init__()
        self.quiet = quiet
        with _muted(quiet):
            self.model = AnySplat.from_pretrained(pretrained_ckpt)
        self.model.eval()
        self.model.requires_grad_(False)

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

    def forward(self, context_images: Tensor):
        """
        Takes a (V, 3, H, W) tensor of images in [0, 1] and returns a
        tuple (poses, intrinsics, gaussians).

        Poses is a (V, 4, 4) tensor of camera-to-world matrices and
        intrinsics a (V, 3, 3) tensor of pinhole matrices, both already
        in the convention that Gaussians.rasterize() expects.
        """
        assert context_images.shape[1] == 3, f"Expected (V, 3, H, W) shape for context images, got {context_images.shape}"
        assert context_images.dim() == 4, f"context_images should have dim 4, got {context_images.dim()}"
        with _muted(self.quiet):
            # The encoder works on batches of scenes, we do one at a time
            encoder_output = self.model.encoder(
                context_images.unsqueeze(0), global_step=0, visualization_dump=None
            )
        poses, intrinsics = self._to_dl3dv_convention(
            encoder_output.pred_context_pose, context_images.shape[-2:]
        )
        gaussians = Gaussians.from_anysplat(encoder_output.gaussians)
        return poses, intrinsics, gaussians