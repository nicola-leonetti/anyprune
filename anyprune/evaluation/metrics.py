"""
Image-space metrics for scoring a render against its ground truth.
"""
import torch
from torch import Tensor


def psnr(rendered: Tensor, target: Tensor) -> Tensor:
    """
    Per-view PSNR, in dB, of a (V, 3, H, W) render against its target.
    Both are expected to be in [0, 1].
    """
    mse = (rendered - target).pow(2).flatten(start_dim=1).mean(dim=1)
    return -10.0 * torch.log10(mse)


__all__ = [
    "psnr",
]
