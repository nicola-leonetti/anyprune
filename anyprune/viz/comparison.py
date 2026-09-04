"""
Figures putting side by side the results of different models.
"""
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from torch import Tensor

from ..evaluation import psnr
from ._common import (
    AXIS_COLOR, BLOCK_GAP, INK, SERIES_COLORS, evenly_spaced, plot_frames,
)


# Assigned to models in the order they are compared in
_MODEL_COLORS = SERIES_COLORS
_AXIS_COLOR = AXIS_COLOR


def _format_count(count: float, _position=None) -> str:
    """
    Tick labels for Gaussian counts, which span from the millions a
    feed-forward model emits down to whatever pruning leaves behind.
    """
    if count >= 1e6:
        return f"{count / 1e6:g}M"
    if count >= 1e3:
        return f"{count / 1e3:g}k"
    return f"{count:g}"


@dataclass
class ModelReconstruction:
    """
    What one model made of a scene, ready to be compared against the
    others.
    """
    name: str
    num_gaussians: int
    self_recon: Tensor
    test_render: Tensor


def _plot_gaussian_counts(axis, reconstructions: Sequence[ModelReconstruction]):
    """
    How many Gaussians each model spent on the scene, written out on the
    bars.
    """
    counts = [reconstruction.num_gaussians for reconstruction in reconstructions]
    positions = list(range(len(reconstructions)))

    axis.barh(
        positions, counts, height=0.4,
        color=[_MODEL_COLORS[position % len(_MODEL_COLORS)] for position in positions],
    )
    for position, count in zip(positions, counts):
        axis.text(count, position, f"  {count:,}", va="center", fontsize=9, color=INK)

    axis.set_yticks(positions)
    axis.set_yticklabels([r.name for r in reconstructions], fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("Gaussians reconstructed", fontsize=9, color=INK)
    # Leave the bars room to be labelled without running off the axis
    axis.set_xlim(0, max(counts) * 1.2)
    axis.xaxis.set_major_formatter(FuncFormatter(_format_count))
    axis.tick_params(labelsize=8, colors=INK, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(_AXIS_COLOR)


def plot_reconstructions(
    context_images: Tensor,
    test_images: Tensor,
    reconstructions: Sequence[ModelReconstruction],
    title: str,
    num_shown: int = 5,
) -> Figure:
    """
    Compare `reconstructions` against the ground truth of a scene.

    The figure carries two blocks of rows, the context views each model
    reconstructed from and the held-out views it did not, each opening
    with the ground truth and followed by one row per model, plus a
    panel counting the Gaussians each model produced. `num_shown` frames
    are drawn per row, sampled evenly along the trajectory, and every
    rendered frame is labelled with its PSNR, the row with their mean.

    `context_images` and `test_images` are (V, 3, H, W) in [0, 1], and
    have to line up with the renders each ModelReconstruction carries.
    """
    assert reconstructions, "Nothing to compare"

    context_shown = evenly_spaced(context_images.shape[0], num_shown)
    test_shown = evenly_spaced(test_images.shape[0], num_shown)

    def model_row(render: Tensor, truth: Tensor, shown: Tensor, name: str):
        """One model's row, scored against the views it is drawn over."""
        scores = psnr(render, truth)
        return render[shown], f"{name}\n{scores.mean():.2f} dB", scores[shown]

    context_rows = [(context_images[context_shown], "context views\nground truth", None)]
    context_rows += [
        model_row(r.self_recon, context_images, context_shown, r.name)
        for r in reconstructions
    ]
    test_rows = [(test_images[test_shown], "held-out views\nground truth", None)]
    test_rows += [
        model_row(r.test_render, test_images, test_shown, r.name)
        for r in reconstructions
    ]
    # An empty row of its own tells the two blocks apart
    rows = context_rows + [None] + test_rows

    num_columns = max(len(context_shown), len(test_shown))
    counts_height = 0.22 + 0.16 * len(reconstructions)
    heights = [BLOCK_GAP if row is None else 1 for row in rows] + [counts_height]
    figure = plt.figure(figsize=(1.8 * num_columns, 1.85 * sum(heights) + 0.6))
    grid = figure.add_gridspec(len(heights), num_columns, height_ratios=heights)

    for row, entry in enumerate(rows):
        if entry is None:
            continue
        frames, label, scores = entry
        axes = [figure.add_subplot(grid[row, column]) for column in range(num_columns)]
        plot_frames(axes, frames, label, scores)
    _plot_gaussian_counts(figure.add_subplot(grid[len(rows), :]), reconstructions)

    figure.suptitle(title)
    figure.tight_layout()
    return figure


__all__ = [
    "ModelReconstruction",
    "plot_reconstructions",
]