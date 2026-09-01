"""
Implementation of a histogram to keep track of the gaussian counts used
at training-time.
"""
from typing import List, Optional

import torch
from torch import Generator


class BudgetHistogram:
    """
    A running count of how many steps have trained on a field of each
    size. This information is used to decide the pruning percentage of 
    the next steps.
    """

    def __init__(self, bucket_size: int = 10_000, max_gaussians: int = 400_000):
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
        self.counts[self.bucket_of(num_gaussians)] += 1

    def summary(self, width: int = 8) -> str:
        """
        Returns a string with a summary of the histogram
        """
        lines = []
        for first in range(0, self.num_buckets, width):
            block = self.counts[first:first + width]
            lines.append(
                f"  {first * self.bucket_size // 1000:>4}k-"
                f"{(first + len(block)) * self.bucket_size // 1000:>4}k "
                + " ".join(f"{count:>4}" for count in block)
            )
        return "\n".join(lines)


__all__ = ["BudgetHistogram"]
