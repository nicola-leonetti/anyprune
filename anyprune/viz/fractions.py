"""
The figure of a pruning sweep over kept fractions: every metric against
the share of the field a method kept, one curve per method, in one row
per metric and one column per context view count, for one half of a
scene. Sized for slides and papers: ~16 in wide at four columns, so
that its text stays >= 8 pt at a paper's 7 in text width.
"""
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ._common import AXIS_COLOR, INK


# What a figure draws unless told otherwise: the key of the metric in a
# record, after the half's prefix, and how its axis is labelled.
_DEFAULT_METRICS = (("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓"))

# Tick labels are made when the figure is drawn, outside the rc
# context, so their size is set on the axes rather than here
_FONT_SIZES = {"font.size": 20, "axes.titlesize": 22, "axes.labelsize": 21, "legend.fontsize": 21}
_TICK_SIZE = 19
_SURFACE = "white"


def _marker_size(marker: Optional[str]) -> float:
    """A star fills much less of its box than the other markers, so it is drawn larger to match them."""
    return 16 if marker == "*" else 8


def plot_fraction_curves(
    summary: Sequence[Mapping],
    block: str,
    context_views: Sequence[int],
    methods: Mapping[str, Tuple[str, str, Optional[str], str]],
    title: str,
    metrics: Sequence[Tuple[str, str]] = _DEFAULT_METRICS,
    reference: Optional[str] = None,
) -> Figure:
    """
    'summary' holds one record per (context_views, method, fraction),
    with the mean of each metric under '<block>_<metric>'. 'methods'
    maps the method of a record to its (label, colour, marker,
    linestyle), in the order of the legend; 'reference' is drawn under
    the others.
    """
    with plt.rc_context(_FONT_SIZES):
        figure, axes = plt.subplots(
            len(metrics), len(context_views), figsize=(4.1 * len(context_views), 3.5 * len(metrics)),
            squeeze=False, sharex=True, sharey="row",
        )
        for col, views in enumerate(context_views):
            for row, (metric, label) in enumerate(metrics):
                axis = axes[row, col]
                for method, (_, color, marker, style) in methods.items():
                    points = sorted(
                        (r["fraction"], r[f"{block}_{metric}"]) for r in summary
                        if r["context_views"] == views and r["method"] == method
                    )
                    if not points:
                        continue
                    x, y = zip(*points)
                    axis.plot(
                        [100 * f for f in x], y, color=color, linestyle=style, linewidth=2.2,
                        marker=marker, markersize=_marker_size(marker), markeredgecolor=_SURFACE, markeredgewidth=1.0,
                        zorder=1 if method == reference else 2,
                    )
                if row == 0:
                    axis.set_title(f"{views} views", color=INK, pad=10)
                if col == 0:
                    axis.set_ylabel(label, color=INK)
                if row == len(metrics) - 1:
                    axis.set_xlabel("kept (%)", color=INK)
                axis.set_xticks([10, 40, 70, 100])
                axis.set_xlim(6, 104)
                axis.grid(True, color=AXIS_COLOR, linewidth=0.8, alpha=0.6)
                axis.set_axisbelow(True)
                axis.tick_params(labelsize=_TICK_SIZE, colors=INK, length=0, pad=6)
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    axis.spines[side].set_color(AXIS_COLOR)
        handles = [
            Line2D([], [], color=color, linestyle=style, marker=marker, linewidth=3,
                   markersize=_marker_size(marker) * 11 / 8,
                   markeredgecolor=_SURFACE, label=label)
            for label, color, marker, style in methods.values()
        ]
        figure.suptitle(title, color=INK, fontsize=23)
        figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.8, w_pad=0.6)
        figure.legend(
            handles=handles, loc="upper center", ncol=len(handles) if len(handles) <= 5 else 4,
            frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 0.0),
            columnspacing=1.4, handlelength=2.2, handletextpad=0.5,
        )
    return figure
