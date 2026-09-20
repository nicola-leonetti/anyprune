"""
Implementation of the histograms that keep track of what the steps of a
run have spent: how many Gaussians each of them trained on, and what a
field of that size costs in loss.
"""
from typing import List, Optional, Sequence

import torch
from torch import Generator


class _BucketedHistogram:
    """
    A count of the steps that trained on a field of each size, over a
    range of Gaussian counts cut into equal buckets. Both histograms
    below keep one and lay their own numbers out over the same buckets.
    """

    def __init__(self, bucket_size: int, max_gaussians: int):
        assert bucket_size > 0, f"Need a positive bucket size, got {bucket_size}"
        assert max_gaussians >= bucket_size, (
            f"Need room for at least one bucket, got a ceiling of "
            f"{max_gaussians} against buckets of {bucket_size}"
        )
        assert max_gaussians % bucket_size == 0, (
            f"The buckets have to divide the range evenly, and "
            f"{bucket_size} does not divide {max_gaussians}"
        )
        self.bucket_size = bucket_size
        self.max_gaussians = max_gaussians
        self.counts: List[int] = [0] * (max_gaussians // bucket_size)

    @property
    def num_buckets(self) -> int:
        return len(self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def spread(self) -> float:
        """
        How far from uniform the histogram currently is, as the standard
        deviation of the counts over their mean. Zero is flat.
        """
        if self.total == 0:
            return 0.0
        mean = self.total / self.num_buckets
        variance = sum(
            (count - mean) ** 2 for count in self.counts
        ) / self.num_buckets
        return variance ** 0.5 / mean

    def bucket_of(self, num_gaussians: int) -> int:
        """
        Which bucket a field of this size falls in, with everything
        above the top of the range folded into the top bucket.
        """
        assert num_gaussians >= 0, f"Got a field of {num_gaussians} Gaussians"
        return min(num_gaussians // self.bucket_size, self.num_buckets - 1)

    def _count(self, num_gaussians: int) -> int:
        """Count a step that trained on a field of this size."""
        bucket = self.bucket_of(num_gaussians)
        self.counts[bucket] += 1
        return bucket

    def _rows(self, cells: Sequence[str], width: int) -> str:
        """
        Lay one already written cell per bucket out in blocks of
        'width', each labelled with the range of sizes it covers.
        """
        assert len(cells) == self.num_buckets, (
            f"Need a cell per bucket, got {len(cells)} for {self.num_buckets}"
        )
        lines = []
        for first in range(0, self.num_buckets, width):
            block = cells[first:first + width]
            lines.append(
                f"  {first * self.bucket_size // 1000:>4}k-"
                f"{(first + len(block)) * self.bucket_size // 1000:>4}k "
                + " ".join(block)
            )
        return "\n".join(lines)


class BudgetHistogram(_BucketedHistogram):
    """
    A running count of how many steps have trained on a field of each
    size. This information is used to decide the pruning percentage of 
    the next steps.
    """

    def __init__(self, bucket_size: int = 10_000, max_gaussians: int = 400_000):
        super().__init__(bucket_size, max_gaussians)

    def choose(
        self,
        num_predicted: int,
        max_gaussians: Optional[int] = None,
        generator: Optional[Generator] = None,
    ) -> int:
        """
        Pick how many of 'num_predicted' Gaussians this step should
        keep, held under 'max_gaussians' if the card has a ceiling
        lower than the histogram's range.
        """
        assert num_predicted > 0, f"Nothing to thin, got {num_predicted}"
        ceiling = num_predicted if max_gaussians is None else min(
            num_predicted, max_gaussians
        )
        assert ceiling > 0, f"No room to put anything in, got {max_gaussians}"
        # Every bucket from the bottom up to the one the untouched
        # prediction lands in, which is the last that can be reached.
        # Its own bucket counts: a step is allowed to keep nearly all of
        # what it predicted when that is the size the run is short of.
        reachable = self.bucket_of(ceiling) + 1
        fewest = min(self.counts[:reachable])
        tied = [
            bucket for bucket in range(reachable)
            if self.counts[bucket] == fewest
        ]
        chosen = tied[
            torch.randint(len(tied), (1,), generator=generator).item()
        ]
        low = max(chosen * self.bucket_size, 1)
        high = min((chosen + 1) * self.bucket_size - 1, ceiling)
        return low + torch.randint(
            high - low + 1, (1,), generator=generator
        ).item()

    def record(self, num_gaussians: int) -> None:
        """Count a step that trained on a field of this size."""
        self._count(num_gaussians)

    def summary(self, width: int = 8) -> str:
        """
        Returns a string with a summary of the histogram
        """
        return self._rows([f"{count:>4}" for count in self.counts], width)


class LossHistogram(_BucketedHistogram):
    """
    A running mean of the loss the steps at each field size came back
    with, kept so that a step can be weighed against what its own budget
    usually costs rather than against every other budget's losses.

    A sparse field leaves more error behind than a dense one whatever
    the network does with it, so its loss is larger, and so is the pull
    it has on the weights: a run that draws its sizes evenly is still
    spending its gradient unevenly. Dividing a step's loss by the mean
    of its bucket over the mean of every step takes that difference back
    out. It is a ratio rather than a difference so that the terms keep
    their own scale, and it is against the run's own mean so that the
    average step is left where it was, which is what lets a run that
    normalizes keep the learning rate and the clipping norm of one that
    does not.

    Nothing here reads the budget histogram or is read by it: this one
    only ever sees the size a step ended up training on, however that
    size was arrived at.
    """

    def __init__(
        self,
        bucket_size: int = 10_000,
        max_gaussians: int = 500_000,
        momentum: float = 0.9,
        max_ratio: float = 10.0,
    ):
        super().__init__(bucket_size, max_gaussians)
        assert 0.0 <= momentum < 1.0, (
            f"Need a momentum in [0, 1), got {momentum}"
        )
        assert max_ratio >= 1.0, (
            f"Need to allow a bucket at least the run's own mean, got a "
            f"ceiling of {max_ratio} on the ratio"
        )
        self.momentum = momentum
        self.max_ratio = max_ratio
        self.means: List[Optional[float]] = [None] * self.num_buckets
        self.mean: Optional[float] = None
        # A bucket is written to only by the steps that land in it, one
        # step in num_buckets when the sizes are drawn evenly, while the
        # mean over everything is written to by all of them. Slowing the
        # second down by that same factor leaves the two looking equally
        # far back over the run, which is what makes their ratio a
        # statement about the size rather than about the last few scenes.
        self.total_momentum = 1.0 - (1.0 - momentum) / self.num_buckets

    @staticmethod
    def _folded(
        mean: Optional[float], value: float, momentum: float
    ) -> float:
        """
        The mean with one more observation in it, which is the
        observation itself when there was no mean yet.
        """
        return value if mean is None else momentum * mean + (1 - momentum) * value

    def _nearest_mean(self, bucket: int) -> Optional[float]:
        """
        What this bucket costs, or the closest visited bucket's mean
        when this one has not been trained on yet, so that the first
        step at a size is weighed by what the sizes around it cost
        rather than not weighed at all.
        """
        for offset in range(self.num_buckets):
            for side in (bucket - offset, bucket + offset):
                if 0 <= side < self.num_buckets and self.means[side] is not None:
                    return self.means[side]
        return None

    def scale_of(self, num_gaussians: int) -> float:
        """
        How much of the run's typical loss a field of this size usually
        costs, which is what a step's loss is to be divided by, held
        inside a factor of 'max_ratio' either way.

        Comes back as 1.0 until the run has trained on something, which
        leaves the very first step scored as it arrived.
        """
        if self.mean is None or self.mean <= 0.0:
            return 1.0
        mean = self._nearest_mean(self.bucket_of(num_gaussians))
        if mean is None or mean <= 0.0:
            return 1.0
        return min(max(mean / self.mean, 1.0 / self.max_ratio), self.max_ratio)

    def record(self, num_gaussians: int, loss: float) -> None:
        """
        Fold the loss a step on a field of this size came back with into
        that size's mean and into the run's.
        """
        bucket = self._count(num_gaussians)
        self.means[bucket] = self._folded(
            self.means[bucket], loss, self.momentum
        )
        self.mean = self._folded(self.mean, loss, self.total_momentum)

    def summary(self, width: int = 8) -> str:
        """
        Returns a string with the mean loss of every bucket the run has
        trained on, blank where it has not trained at all.
        """
        return self._rows(
            [
                f"{'':>6}" if mean is None else f"{mean:>6.4f}"
                for mean in self.means
            ],
            width,
        )


__all__ = ["BudgetHistogram", "LossHistogram"]
