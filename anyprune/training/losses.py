"""
Definition of the losses.
"""
from typing import Dict, Tuple

import torch.nn as nn
from torch import Tensor

from ..models.utils import lpips_loss_fn


class PhotometricLoss(nn.Module):
    """
    SplatFormer's training loss: the mean absolute error between a
    render and its ground truth, optionally a squared-error term and a
    VGG LPIPS term, all averaged over the views.
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.0,
        lpips_weight: float = 1.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.lpips_weight = lpips_weight
        # SplatFormer's own wrapper around lpips, which puts the network
        # on the GPU and freezes it
        self.lpips = lpips_loss_fn() if lpips_weight > 0 else None

    def forward(
        self, rendered: Tensor, target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Score a (V, 3, H, W) render against its target, both in [0, 1],
        and return the total loss along with each term for logging.
        """
        assert rendered.shape == target.shape, (
            f"Rendered {tuple(rendered.shape)} against target {tuple(target.shape)}"
        )
        residual = rendered - target
        terms = {}
        if self.l1_weight > 0:
            terms["l1"] = residual.abs().mean() * self.l1_weight
        if self.l2_weight > 0:
            terms["l2"] = residual.pow(2).mean() * self.l2_weight
        assert terms or self.lpips is not None, "The loss has no terms"
        if self.lpips is not None:
            # A view at a time, as SplatFormer does it: VGG features for
            # a whole set of views at once are one of the larger things
            # a step allocates. lpips_loss_fn() takes channels last.
            lpips = sum(
                self.lpips(view.permute(1, 2, 0)[None], truth.permute(1, 2, 0)[None]).mean()
                for view, truth in zip(rendered, target)
            )
            terms["lpips"] = lpips / rendered.shape[0] * self.lpips_weight
        total = sum(terms.values())
        return total, {name: term.item() for name, term in terms.items()}


__all__ = [
    "PhotometricLoss",
]
