"""
The pieces the figures in this package are built out of.
"""
from typing import Optional, Sequence

import torch
from torch import Tensor


INK = "#52514e"
AXIS_COLOR = "#c9c8c3"

# Handed out to whatever a figure puts side by side, models or budgets,
# in this fixed order and never cycled, so that a series keeps its color
# as others join it.
SERIES_COLORS = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)

# How much room to leave between two blocks of rows, as a fraction of
# the height of a row.
BLOCK_GAP = 0.45


def evenly_spaced(num_frames: int, num_shown: int) -> Tensor:
    """
    Indices of `num_shown` frames spread evenly over a trajectory of
    `num_frames`, endpoints included.
    """
    return torch.linspace(0, num_frames - 1, min(num_shown, num_frames)).round().long()


def _annotate(axis, text: str, at_top: bool):
    """
    Write one line on a frame, in a box that keeps it readable whatever
    the frame happens to be of.
    """
    axis.text(
        0.035, 0.965 if at_top else 0.035, text,
        transform=axis.transAxes, ha="left", va="top" if at_top else "bottom",
        fontsize=8, color=INK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="none", alpha=0.75),
    )


def plot_frames(
    axes,
    frames: Tensor,
    label: str,
    scores: Optional[Tensor] = None,
    captions: Optional[Sequence[str]] = None,
):
    """
    Draw a (V, 3, H, W) tensor along a row of axes, writing each frame's
    caption at its top and its PSNR against the ground truth at its
    bottom, whenever there is one of either to write.
    """
    for axis, frame in zip(axes, frames):
        axis.imshow(frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(frames):]:
        axis.set_visible(False)
    axes[0].set_ylabel(label, fontsize=9)

    if captions is not None:
        for axis, caption in zip(axes, captions):
            _annotate(axis, caption, at_top=True)
    if scores is not None:
        for axis, score in zip(axes, scores):
            _annotate(axis, f"{score:.2f} dB", at_top=False)
