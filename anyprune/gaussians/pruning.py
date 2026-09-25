"""
The rules by which a field is cut down to a budget: the uniform draw
this project prunes with, and the importance scores of the pruning
literature, each paired with the way its paper picks a subset out of it.

A rule answers with an order rather than with a field, best kept first,
so that every budget of a column is a prefix of the same order and the
budgets are nested: what a wide budget keeps, a narrow one keeps the top
of. That is how the evaluation can measure one order per scene and read
every budget off it.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Generator, Tensor

from .gaussians import Gaussians
from .importance import BlendingWeights, SCORES, blending_weights, score_of


# How a subset is picked out of a score:
#   - 'random' ignores the score and shuffles, which is this project's
#     own baseline and the only thing the uniform score can do
#   - 'top-k' keeps the highest scores, which is RadSplat's threshold
#     written as a budget: the same order, cut where the budget falls
#     rather than where the threshold does. It is LightGaussian's cut
#     as it stands, which drops a share of the field from the bottom
#   - 'weighted' draws without replacement with probability
#     proportional to the score, which is what Mini-Splatting does, and
#     is not the same thing as the top of it: it keeps some of the low
#     scores and so keeps covering the parts of a scene that no Gaussian
#     is individually important to. Its 'temperature' sharpens or
#     flattens that draw: the probability goes with the score to the
#     power of one over it, so that at 1 it is Mini-Splatting's own
#     draw, towards 0 it becomes 'top-k', and far above 1 it forgets the
#     score and becomes 'random'
SELECTIONS = ("random", "top-k", "weighted")


@dataclass(frozen=True)
class Pruner:
    """
    One way of choosing which Gaussians a budget keeps: a name to report
    it under, the score it reads, and how it cuts that score down.
    """
    name: str
    score: str = "uniform"
    selection: str = "random"
    temperature: float = 1.0
    compensate: Optional[bool] = None

    def __post_init__(self):
        assert self.temperature > 0.0, (
            f"The temperature of a weighted draw has to be positive: "
            f"{self.name!r} asks for {self.temperature}"
        )
        assert self.score in SCORES, (
            f"The score has to be one of {SCORES}: {self.score} is not"
        )
        assert self.selection in SELECTIONS, (
            f"The selection has to be one of {SELECTIONS}: "
            f"{self.selection} is not"
        )
        assert (self.score == "uniform") == (self.selection == "random"), (
            "A uniform score can only be drawn at random, and a measured "
            f"one is never drawn at random: {self.name!r} asks for "
            f"{self.selection!r} out of {self.score!r}"
        )

    @property
    def measured(self) -> bool:
        """Whether this rule has to render the field before it can rank it."""
        return self.score != "uniform"

    @property
    def slug(self) -> str:
        """The name as it goes into a wandb key or a file name."""
        kept = "".join(
            char if char.isalnum() else "-" for char in self.name.lower()
        )
        return "-".join(word for word in kept.split("-") if word)

    def compensates(self, default: bool) -> bool:
        """
        Whether a field this rule thinned has its optical depth put back
        before anything reads it, which follows the run's setting unless
        this rule overrides it.
        """
        return default if self.compensate is None else self.compensate

    @torch.no_grad()
    def order(
        self,
        gaussians: Gaussians,
        poses: Tensor,
        intrinsics: Tensor,
        image_shape: Tuple[int, int],
        generator: Optional[Generator] = None,
        measured: Optional[BlendingWeights] = None,
        **kwargs,
    ) -> Tensor:
        """
        Rank every Gaussian of a field, best first, over the views the
        reconstructor was given, which are handed over the way
        Gaussians.rasterize() takes them.

        'generator' seeds the draws, and is expected to be a CPU one, so
        that a caller that seeds a scene rather than the device gets the
        same order out of the same scene on any machine.

        'measured' is a sweep over those same views that the caller has
        already made, which saves rendering the field again: every rule
        that reads a score reads it off the same five numbers, so a
        caller ranking one field under several rules measures it once.
        """
        num_gaussians = gaussians.num_gaussians
        if self.selection == "random":
            return torch.randperm(
                num_gaussians, generator=generator
            ).to(gaussians.device)

        if measured is None:
            measured = blending_weights(
                gaussians, poses, intrinsics, image_shape, **kwargs
            )
        assert measured.num_gaussians == num_gaussians, (
            f"The measurement is of {measured.num_gaussians:,} Gaussians "
            f"and the field holds {num_gaussians:,}"
        )
        keys = score_of(self.score, measured, gaussians)
        if self.selection == "weighted":
            # Drawing without replacement with probability proportional
            # to the score, all at once: perturbing each log score by a
            # Gumbel and sorting is the same draw as taking one Gaussian
            # at a time and renormalizing what is left, and unlike that
            # loop it comes back as the whole order rather than as one
            # budget's worth of it. A zero score is never drawn, and
            # lands below everything that was. The temperature divides
            # the log score, which is the same as raising the score to
            # one over it before the draw.
            noise = torch.rand(
                num_gaussians, generator=generator
            ).to(keys.device).clamp(min=torch.finfo(keys.dtype).tiny)
            keys = keys.log() / self.temperature - (-noise.log()).log()

        # Ties broken at random rather than by the order the
        # reconstructor happened to predict them in, which in a
        # feed-forward field is the order of the pixels they were
        # predicted from. Every score hands out a great many zeros -
        # everything no ray ever stopped on - and a budget wider than
        # what scored above zero would otherwise be filled with one
        # corner of the image.
        shuffled = torch.randperm(
            num_gaussians, generator=generator
        ).to(keys.device)
        return shuffled[torch.argsort(keys[shuffled], descending=True, stable=True)]

    @torch.no_grad()
    def prune(
        self,
        gaussians: Gaussians,
        num_gaussians: int,
        poses: Tensor,
        intrinsics: Tensor,
        image_shape: Tuple[int, int],
        generator: Optional[Generator] = None,
        measured: Optional[BlendingWeights] = None,
        **kwargs,
    ) -> Gaussians:
        """
        The 'num_gaussians' of a field this rule keeps, or the field
        itself when it is already that small.
        """
        if gaussians.num_gaussians <= num_gaussians:
            return gaussians
        kept = self.order(
            gaussians, poses, intrinsics, image_shape,
            generator=generator, measured=measured, **kwargs,
        )
        return gaussians[kept[:num_gaussians]]


__all__ = [
    "Pruner",
    "SELECTIONS",
]
