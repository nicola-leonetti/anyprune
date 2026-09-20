"""
The figure of the evaluation protocol: the PSNR of every pruning method
as a bar, grouped by Gaussian budget, in one panel per generator and
context view count, with the two halves of a scene as the two rows of
a panel; or several metrics, each with its two rows.
"""
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ._common import AXIS_COLOR, INK


# The two halves of a scene, keyed the way the records key them.
_BLOCKS = (("self", "self-reconstruction"), ("nvs", "novel views"))

# What a figure draws unless told otherwise: the key of the metric in a
# record, after the half's prefix, and how its axis is labelled.
_DEFAULT_METRICS = (("psnr", "PSNR (dB)"),)

# The share of a budget's slot its bars fill, the rest being air.
_GROUP_WIDTH = 0.8


def _format_budget(budget: int) -> str:
    """A budget as a tick label, e.g. 250k."""
    return f"{budget / 1e6:g}M" if budget >= 1e6 else f"{budget / 1e3:g}k"


def _format_count(count: float) -> str:
    """A Gaussian count as a bar's annotation, e.g. 53.5k."""
    return f"{count / 1e6:.2f}M" if count >= 1e6 else f"{count / 1e3:.1f}k"


def _style_axis(axis, ticks: Sequence[str]) -> None:
    axis.set_xticks(range(len(ticks)))
    axis.set_xticklabels(ticks)
    axis.tick_params(labelsize=8, colors=INK, length=0, labelleft=True)
    axis.grid(True, axis="y", color=AXIS_COLOR, linewidth=0.6, alpha=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS_COLOR)


def plot_eval_histogram(
    summary: Sequence[Mapping],
    generators: Sequence[str],
    context_views: Sequence[int],
    budgets: Sequence[int],
    methods: Sequence[Tuple[str, str, Optional[str]]],
    title: str,
    metrics: Sequence[Tuple[str, str]] = _DEFAULT_METRICS,
    annotate_counts: bool = False,
) -> Figure:
    """
    'summary' holds one record per (generator, context_views, method,
    budget) with its 'num_gaussians', 'predicted_gaussians' and, for
    each of 'metrics' as (key, axis label), its 'self_<key>' and
    'nvs_<key>'. 'methods' names the bars of a group, in order, as
    (label, color, hatch). A generator gets two rows per metric, one per
    half of the scene.

    A budget's tick carries the number of Gaussians actually kept when
    the budget was wider than what the generator predicted; the
    prediction itself is in the panel's title. With 'annotate_counts'
    every bar also carries the count its own method kept, for cells
    whose methods do not keep the same number.
    """
    assert generators and context_views and budgets and methods, "Nothing to plot"
    assert metrics, "Nothing to plot the methods by"
    cells = {
        (r["generator"], r["context_views"], r["method"], r["budget"]): r
        for r in summary
    }
    blocks = [
        (f"{block}_{key}", f"{block_name}\n{label}")
        for key, label in metrics for block, block_name in _BLOCKS
    ]
    rows, columns = len(blocks) * len(generators), len(context_views)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.6 * columns, 2.7 * rows),
        sharey="row", squeeze=False,
    )
    width = _GROUP_WIDTH / len(methods)

    for g, generator in enumerate(generators):
        for c, views in enumerate(context_views):
            panel = [
                cells[(generator, views, label, budget)]
                for label, _, _ in methods for budget in budgets
                if (generator, views, label, budget) in cells
            ]
            ticks = []
            for budget in budgets:
                kept = min(
                    (r["num_gaussians"] for r in panel if r["budget"] == budget),
                    default=budget,
                )
                ticks.append(
                    _format_budget(budget)
                    + (f"\n({kept:,.0f})" if kept < budget else "")
                )
            predicted = panel[0]["predicted_gaussians"] if panel else None
            for b, (key, axis_label) in enumerate(blocks):
                axis = axes[len(blocks) * g + b][c]
                for m, (label, color, hatch) in enumerate(methods):
                    offset = (m - (len(methods) - 1) / 2) * width
                    drawn = [
                        (i + offset, cells[(generator, views, label, budget)])
                        for i, budget in enumerate(budgets)
                        if (generator, views, label, budget) in cells
                    ]
                    if not drawn:
                        continue
                    axis.bar(
                        [x for x, _ in drawn], [r[key] for _, r in drawn], width,
                        color=color, hatch=hatch, edgecolor="white",
                        linewidth=0.6,
                    )
                    if annotate_counts:
                        for x, r in drawn:
                            axis.annotate(
                                _format_count(r["num_gaussians"]), (x, r[key]),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=6,
                                rotation=90, color=INK,
                            )
                _style_axis(axis, ticks)
                if c == 0:
                    axis.set_ylabel(axis_label, fontsize=9, color=INK)
                if b == 0:
                    axis.set_title(
                        f"{generator}, {views} context views"
                        + ("" if predicted is None else f"\n({predicted:,.0f} Gaussians)"),
                        fontsize=10, color=INK, pad=6,
                    )

    # The limits of a row once all of its panels are drawn, from zero to
    # the tallest bar of any of them, with room over it for the
    # annotations: set on one panel, since the row shares its y axis,
    # and only now, because fixing a limit turns autoscaling off and a
    # later panel's taller bars would otherwise run off the top
    for row in axes:
        tallest = max(axis.dataLim.ymax for axis in row)
        row[0].set_ylim(0, tallest * (1.25 if annotate_counts else 1.05))

    handles = [
        Patch(facecolor=color, hatch=hatch, edgecolor="white", label=label)
        for label, color, hatch in methods
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 5),
        frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(title, fontsize=12, color=INK)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    return figure


__all__ = [
    "plot_eval_histogram",
]
