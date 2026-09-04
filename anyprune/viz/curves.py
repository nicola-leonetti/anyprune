"""
Figures summarizing a whole evaluation sweep rather than one scene.
"""
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ._common import AXIS_COLOR, INK, SERIES_COLORS


# The two halves of a scene a cell is scored on, keyed the way the sweep
# keys them and named the way a panel can carry.
_BLOCKS = (
    ("self", "self-reconstruction, on the views the reconstructor saw"),
    ("nvs", "novel view synthesis, on the views held out from it"),
)

# The metrics, in the order the report's tables put them, each with the
# arrow that says which way is better.
_METRICS = (
    ("psnr", "PSNR (dB) ↑"),
    ("ssim", "SSIM ↑"),
    ("lpips", "LPIPS ↓"),
)

# What is drawn of each budget: the refined Gaussians solid, and the
# thinned ones they were refined from dashed underneath, so that a panel
# reads as the distance between the two rather than as one line whose
# height means nothing on its own.
_STAGES = (
    ("refined", "refined", "-", 1.0),
    ("input", "thinned", "--", 0.5),
)


def _format_budget(budget: int) -> str:
    """A budget as a legend entry, e.g. 250k."""
    return f"{budget / 1e6:g}M" if budget >= 1e6 else f"{budget / 1e3:g}k"


def plot_eval_curves(
    scores: Mapping[Tuple[int, int], Mapping[str, float]],
    context_views: Sequence[int],
    budgets: Sequence[int],
    title: str,
) -> Figure:
    """
    Draw what a sweep measured: each metric against the number of
    context views the scene was reconstructed from, one line per
    Gaussian budget, over both halves of every scene.

    'scores' is the sweep keyed by (context views, budget), each cell
    holding one number per '<block>/<metric>/<stage>' out of _BLOCKS,
    _METRICS and _STAGES. A cell that is missing, because no scene fit
    at it, is a gap in its line rather than a hole in the figure.

    The view counts are laid out on a log axis, since they double their
    way up the protocol and would otherwise crowd into the left of every
    panel, and are labelled with the counts themselves rather than with
    powers of two.
    """
    assert context_views and budgets, "Nothing to plot"

    # One y scale per metric, shared down the column. The two blocks sit
    # within a decibel of each other, so scaling each row to its own
    # range rescales that gap away and draws the two rows as the same
    # picture; on a shared scale the rows can actually be read against
    # each other, which is the whole reason they are stacked.
    figure, axes = plt.subplots(
        len(_BLOCKS), len(_METRICS),
        figsize=(4.2 * len(_METRICS), 3.3 * len(_BLOCKS)),
        sharex=True, sharey="col", squeeze=False,
    )
    for row, (block, block_name) in enumerate(_BLOCKS):
        for column, (metric, metric_name) in enumerate(_METRICS):
            axis = axes[row][column]
            for position, budget in enumerate(budgets):
                color = SERIES_COLORS[position % len(SERIES_COLORS)]
                for stage, _, style, alpha in _STAGES:
                    key = f"{block}/{metric}/{stage}"
                    drawn = [
                        (views, scores[(views, budget)][key])
                        for views in context_views
                        if key in scores.get((views, budget), {})
                    ]
                    if not drawn:
                        continue
                    axis.plot(
                        [views for views, _ in drawn],
                        [value for _, value in drawn],
                        linestyle=style, color=color, alpha=alpha,
                        linewidth=1.7, marker="o", markersize=3.5,
                    )
            axis.set_ylabel(metric_name, fontsize=9, color=INK)
            axis.set_xscale("log", base=2)
            axis.set_xticks(list(context_views))
            axis.set_xticklabels([str(views) for views in context_views])
            axis.minorticks_off()
            axis.tick_params(labelsize=8, colors=INK, length=3, labelleft=True)
            axis.grid(True, axis="y", color=AXIS_COLOR, linewidth=0.6, alpha=0.5)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color(AXIS_COLOR)
            if row == len(_BLOCKS) - 1:
                axis.set_xlabel("context views", fontsize=9, color=INK)
        # The half of the scene the row is of, written once over the row
        # rather than onto each of its three panels
        axes[row][len(_METRICS) // 2].set_title(
            block_name, fontsize=10, color=INK, pad=8,
        )

    handles = [
        Line2D(
            [], [], color=SERIES_COLORS[position % len(SERIES_COLORS)],
            linewidth=1.7, marker="o", markersize=3.5,
            label=f"{_format_budget(budget)} Gaussians",
        )
        for position, budget in enumerate(budgets)
    ] + [
        Line2D([], [], color=INK, linestyle=style, alpha=alpha, linewidth=1.7, label=label)
        for _, label, style, alpha in _STAGES
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 6),
        frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(title, fontsize=12, color=INK)
    # Room at the foot for the legend, which sits under the panels
    # rather than inside one of them: the lines run across the whole
    # width and there is no corner of a panel it would not cover
    figure.tight_layout(rect=(0, 0.07, 1, 0.97))
    return figure


__all__ = [
    "plot_eval_curves",
]
