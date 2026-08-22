"""
Figures putting side by side the results of different models.
"""
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from torch import Tensor


# Assigned to models in this fixed order, never cycled, so that a model
# keeps its color as others join the comparison.
_MODEL_COLORS = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
_INK = "#52514e"
_AXIS_COLOR = "#c9c8c3"


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


def _evenly_spaced(num_frames: int, num_shown: int) -> Tensor:
    """
    Indices of `num_shown` frames spread evenly over a trajectory of
    `num_frames`, endpoints included.
    """
    return torch.linspace(0, num_frames - 1, min(num_shown, num_frames)).round().long()


def _plot_frames(axes, frames: Tensor, label: str):
    """Draw a (V, 3, H, W) tensor along a row of axes."""
    for axis, frame in zip(axes, frames):
        axis.imshow(frame.permute(1, 2, 0).cpu().numpy())
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(frames):]:
        axis.set_visible(False)
    axes[0].set_ylabel(label, fontsize=9)


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
        axis.text(count, position, f"  {count:,}", va="center", fontsize=9, color=_INK)

    axis.set_yticks(positions)
    axis.set_yticklabels([r.name for r in reconstructions], fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("Gaussians reconstructed", fontsize=9, color=_INK)
    # Leave the bars room to be labelled without running off the axis
    axis.set_xlim(0, max(counts) * 1.2)
    axis.xaxis.set_major_formatter(FuncFormatter(_format_count))
    axis.tick_params(labelsize=8, colors=_INK, length=0)
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
    are drawn per row, sampled evenly along the trajectory.

    `context_images` and `test_images` are (V, 3, H, W) in [0, 1], and
    have to line up with the renders each ModelReconstruction carries.
    """
    assert reconstructions, "Nothing to compare"

    context_shown = _evenly_spaced(context_images.shape[0], num_shown)
    test_shown = _evenly_spaced(test_images.shape[0], num_shown)

    rows = [(context_images[context_shown], "context views\nground truth")]
    rows += [
        (r.self_recon[context_shown], r.name) for r in reconstructions
    ]
    rows += [(test_images[test_shown], "held-out views\nground truth")]
    rows += [
        (r.test_render[test_shown], r.name) for r in reconstructions
    ]

    num_columns = max(len(context_shown), len(test_shown))
    counts_height = 0.22 + 0.16 * len(reconstructions)
    figure = plt.figure(figsize=(1.8 * num_columns, 1.85 * (len(rows) + counts_height) + 0.6))
    grid = figure.add_gridspec(
        len(rows) + 1, num_columns, height_ratios=[1] * len(rows) + [counts_height]
    )

    for row, (frames, label) in enumerate(rows):
        axes = [figure.add_subplot(grid[row, column]) for column in range(num_columns)]
        _plot_frames(axes, frames, label)
    _plot_gaussian_counts(figure.add_subplot(grid[len(rows), :]), reconstructions)

    figure.suptitle(title)
    figure.tight_layout()
    return figure


__all__ = [
    "ModelReconstruction",
    "plot_reconstructions",
]