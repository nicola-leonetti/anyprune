"""
A PyTorch module to run AnySplat to produce Gaussians from a set of 
views, all with a simplified interface.
"""
import torch
import torch.nn as nn

from .utils import _muted
with _muted(True): from ..models import AnySplat


class FrozenAnySplat(nn.Module):
    def __init__(self, pretrained_ckpt: str, quiet: bool):
        super().__init__()
        self.quiet = quiet

        with _muted(quiet):
            self.model = AnySplat.from_pretrained(pretrained_ckpt)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward():
        pass