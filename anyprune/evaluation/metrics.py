"""
Image-space metrics for scoring a render against its ground truth.

Every metric here takes a (V, 3, H, W) render and its target, both in
[0, 1], and answers per view rather than over the set, so that a caller
can average over the views it cares about instead of being handed a
number it cannot take apart again.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.utils import lpips_loss_fn


# SSIM as the Gaussian splatting literature measures it, and SplatFormer
# with it: an 11-pixel Gaussian window of sigma 1.5, with the two
# stabilizing constants of the original definition written for a
# dynamic range of 1.
_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_C1 = 0.01 ** 2
_SSIM_C2 = 0.03 ** 2

# How many views VGG is shown at once. LPIPS holds the activations of
# every one of its layers for every view it is handed, which is one of
# the larger things a scoring pass allocates, so the views go through it
# in blocks rather than all at once.
_LPIPS_VIEWS_PER_PASS = 8


def psnr(rendered: Tensor, target: Tensor) -> Tensor:
    """
    Per-view PSNR, in dB, of a (V, 3, H, W) render against its target.
    Both are expected to be in [0, 1]. Higher is closer.
    """
    mse = (rendered - target).pow(2).flatten(start_dim=1).mean(dim=1)
    return -10.0 * torch.log10(mse)


@lru_cache(maxsize=None)
def _ssim_window(
    channels: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """
    The Gaussian window SSIM averages through, written as the (C, 1, W,
    W) kernel of one depthwise convolution.

    Cached because it is the same handful of numbers at every call, and
    building it per view set puts a host-to-device copy in the middle of
    every score.
    """
    offsets = torch.arange(_SSIM_WINDOW, dtype=torch.float64) - _SSIM_WINDOW // 2
    line = torch.exp(-offsets.pow(2) / (2 * _SSIM_SIGMA ** 2))
    line = line / line.sum()
    window = torch.outer(line, line).to(device=device, dtype=dtype)
    return window.expand(channels, 1, _SSIM_WINDOW, _SSIM_WINDOW).contiguous()


def ssim(rendered: Tensor, target: Tensor) -> Tensor:
    """
    Per-view SSIM of a (V, 3, H, W) render against its target, both in
    [0, 1], averaged over the pixels and the channels of each view.
    Higher is closer, and 1 is the same image.
    """
    assert rendered.shape == target.shape, (
        f"Rendered {tuple(rendered.shape)} against target {tuple(target.shape)}"
    )
    channels = rendered.shape[-3]
    window = _ssim_window(channels, rendered.device, rendered.dtype)
    padding = _SSIM_WINDOW // 2

    def blur(images: Tensor) -> Tensor:
        return F.conv2d(images, window, padding=padding, groups=channels)

    mean_rendered, mean_target = blur(rendered), blur(target)
    mean_rendered_sq, mean_target_sq = mean_rendered.pow(2), mean_target.pow(2)
    mean_cross = mean_rendered * mean_target
    # The second moments, each written around the local mean the window
    # just measured rather than around zero
    var_rendered = blur(rendered * rendered) - mean_rendered_sq
    var_target = blur(target * target) - mean_target_sq
    covariance = blur(rendered * target) - mean_cross

    similarity = (
        (2 * mean_cross + _SSIM_C1) * (2 * covariance + _SSIM_C2)
    ) / (
        (mean_rendered_sq + mean_target_sq + _SSIM_C1)
        * (var_rendered + var_target + _SSIM_C2)
    )
    return similarity.flatten(start_dim=1).mean(dim=1)


class LPIPS:
    """
    Perceptual distance through VGG, the same network and the same
    weights the training loss is taken through, held open across calls
    so that the weights are only read once.

    Built by the caller rather than at import: it is a VGG on the card,
    and the two metrics above are used in places that never ask for one.
    """

    def __init__(self, views_per_pass: int = _LPIPS_VIEWS_PER_PASS):
        # SplatFormer's own wrapper, which puts the network on the GPU
        # and freezes it
        self.lpips = lpips_loss_fn()
        self.views_per_pass = views_per_pass

    @torch.no_grad()
    def __call__(self, rendered: Tensor, target: Tensor) -> Tensor:
        """
        Per-view LPIPS of a (V, 3, H, W) render against its target, both
        in [0, 1]. Lower is closer, and 0 is the same image.
        """
        assert rendered.shape == target.shape, (
            f"Rendered {tuple(rendered.shape)} against target {tuple(target.shape)}"
        )
        # lpips_loss_fn() takes its images channels last and answers
        # (V, 1, 1, 1), one number per view
        scores = [
            self.lpips(
                rendered[first:first + self.views_per_pass].permute(0, 2, 3, 1),
                target[first:first + self.views_per_pass].permute(0, 2, 3, 1),
            ).flatten()
            for first in range(0, rendered.shape[0], self.views_per_pass)
        ]
        return torch.cat(scores)


__all__ = [
    "LPIPS",
    "psnr",
    "ssim",
]
