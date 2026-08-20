"""
A PyTorch module to run AnySplat to produce Gaussians from a set of 
views, all with a simplified interface.
"""
import torch
import torch.nn as nn
from torch import Tensor

from anyprune.utils import _muted
with _muted(True): from ..models import AnySplat


class FrozenAnySplat(nn.Module):
    def __init__(self, pretrained_ckpt: str, quiet: bool):
        super().__init__()
        self.quiet = quiet
        with _muted(quiet):
            self.model = AnySplat.from_pretrained(pretrained_ckpt)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, context_images: Tensor):
        """
        Takes a (V, 3, H, W) tensor of images and returns a tuple 
        (gaussians, poses).
        """
        assert context_images.shape[1] == 3, f"Expected (V, 3, H, W) shape for context images, got {context_images.shape}"
        assert context_images.dim() == 4, f"context_images should have dim 4, got {context_images.dim()}"
        with _muted(self.quiet):
            encoder_output = self.model.encoder(
                context_images, global_step=0, visualization_dump=None
            )
        poses = enc_out.pred_context_pose
        gaussians = enc_out.gaussians
        return poses, gaussians