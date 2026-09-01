"""
Figures showing what refinement does to a set of thinned Gaussians.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

from ..evaluation import psnr
from ._common import BLOCK_GAP, INK, evenly_spaced, plot_frames


# What the rows of a block are, in the order they are drawn. The renders
# are all of the same reconstruction, whole and then thinned and then
# refined, so the labels say which of those they are rather than naming
# the network. The first is the ground truth and the rest are scored
# against it.
_ROW_LABELS = ("GT", "all Gaussians", "downsampled", "downsampled and refined")

# Room left for the row labels down the side and for the two-line title
# across the top, in inches, on top of what the frames themselves need.
_LABEL_WIDTH = 1.1
_TITLE_HEIGHT = 0.9
# And for the heading each block opens with, which is charged per block
# rather than once: a heading takes its room out of the row under it,
# which would otherwise squeeze every frame in the figure.
_HEADING_HEIGHT = 0.3


@dataclass
class RefinementBlock:
    """
    One set of views of a scene and the renders of it being put against
    each other: the thinned Gaussians a reconstructor's output was cut
    down to, what refinement made of them, and, when there is one, the
    whole prediction they were thinned out of.
    """
    name: str          # what the block is, written above it
    tag: str           # how a view of it is numbered, e.g. 'ctx'
    truth: Tensor      # (V, 3, H, W), in [0, 1]
    downsampled: Tensor
    refined: Tensor
    full: Optional[Tensor] = None

    def rows(self) -> List[Tuple[str, Tensor]]:
        """
        The rows the block is drawn as, labelled and in drawing order,
        leaving out the whole prediction when it was not rendered.
        """
        frames = (self.truth, self.full, self.downsampled, self.refined)
        return [
            (label, row)
            for label, row in zip(_ROW_LABELS, frames) if row is not None
        ]


def plot_refinement(
    blocks: Sequence[RefinementBlock],
    num_gaussians: int,
    num_context_views: int,
    title: str,
    num_shown: int = 4,
    num_input_gaussians: Optional[int] = None,
) -> Figure:
    """
    Show what one set of Gaussians renders as before and after
    refinement, over every block of views it is scored on.

    Each block becomes a row of ground truth and a row per render of it,
    `num_shown` views wide, sampled evenly along the block's views. A
    render is labelled with its PSNR against the ground truth above it,
    per view and, in the row's label, averaged over the whole block
    rather than over the views drawn. Every frame carries where in its
    own block's sequence it sits, since the blocks interleave along the
    capture and which view of which set is being looked at is otherwise
    guesswork.

    How many Gaussians the renders are of and how many context views
    they came out of go in the title: both vary between the figures a
    run produces, and neither can be read off the pixels. Where the
    blocks carry a render of the whole prediction as well,
    `num_input_gaussians` says how many Gaussians that row is of, so the
    row and what the thinning cost against it can be read together.
    """
    assert blocks, "Nothing to plot"

    rows = []
    for position, block in enumerate(blocks):
        if position:
            # A blank row of its own tells the blocks apart
            rows.append(None)
        num_views = block.truth.shape[0]
        shown = evenly_spaced(num_views, num_shown)
        captions = [
            f"{block.tag} {index + 1}/{num_views}" for index in shown.tolist()
        ]
        for label, frames in block.rows():
            scores = None if frames is block.truth else psnr(frames, block.truth)
            rows.append((
                frames[shown],
                label if scores is None else f"{label}\n{scores.mean():.2f} dB",
                None if scores is None else scores[shown],
                captions,
                block.name if frames is block.truth else None,
            ))

    num_columns = min(num_shown, max(block.truth.shape[0] for block in blocks))
    heights = [BLOCK_GAP if row is None else 1 for row in rows]
    # A frame is drawn at its own aspect ratio and will not stretch to
    # fill its cell, so the cells are cut to the frames rather than the
    # other way round: get this wrong and every frame shrinks to the
    # smaller side of its cell and the figure is mostly white.
    frame_height, frame_width = blocks[0].truth.shape[-2:]
    cell = 1.9
    figure = plt.figure(
        figsize=(
            cell * num_columns + _LABEL_WIDTH,
            cell * frame_height / frame_width * sum(heights)
            + _TITLE_HEIGHT + _HEADING_HEIGHT * len(blocks),
        ),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        len(rows), num_columns, height_ratios=heights, hspace=0.02, wspace=0.02,
    )

    for row, entry in enumerate(rows):
        if entry is None:
            continue
        frames, label, scores, captions, heading = entry
        axes = [figure.add_subplot(grid[row, column]) for column in range(num_columns)]
        plot_frames(axes, frames, label, scores, captions)
        if heading is not None:
            axes[0].set_title(
                heading, loc="left", fontsize=10, color=INK, fontweight="bold"
            )

    counts = (
        f"{num_gaussians:,} Gaussians" if num_input_gaussians is None
        else f"{num_gaussians:,} Gaussians of the {num_input_gaussians:,} predicted"
    )
    figure.suptitle(f"{title}\n{counts} from {num_context_views} context views")
    return figure


__all__ = [
    "RefinementBlock",
    "plot_refinement",
]
