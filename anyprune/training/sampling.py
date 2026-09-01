"""
Choosing which frames of a scene a training step reconstructs from and
which it is scored on.
"""
from typing import Optional, Tuple

import torch
from torch import Generator, Tensor


def sample_view_indices(
    num_frames: int,
    num_views: int,
    stride: int = 1,
    generator: Optional[Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Pick 'num_views' consecutive frames out of a random point of a scene 
    of 'num_frames' and split them evenly into context views and 
    held-out test views.

    Returns the frames to read from the scene, and then the two halves
    as indices relative to those frames rather than to all the frames.

    'stride' spreads the same number of views over a longer stretch, 
    widening the baseline between them.
    """
    assert num_views % 2 == 0, f"Need an even number of views, got {num_views}"
    span = (num_views - 1) * stride + 1
    assert span <= num_frames, (
        f"A run of {num_views} views at stride {stride} spans {span} frames, "
        f"but the scene only has {num_frames}"
    )
    first = torch.randint(num_frames - span + 1, (1,), generator=generator).item()
    frames = first + torch.arange(num_views) * stride
    sampled = torch.arange(num_views)
    return frames, sampled[::2], sampled[1::2]


def sample_num_context_views(
    minimum: int,
    maximum: int,
    generator: Optional[Generator] = None,
) -> int:
    """
    Draw how many context views a step reconstructs from, uniformly
    between `minimum` and `maximum` inclusive.
    """
    assert 2 <= minimum <= maximum, (
        f"Need 2 <= minimum <= maximum context views, got {minimum} and {maximum}"
    )
    span = maximum - minimum + 1
    return minimum + torch.randint(span, (1,), generator=generator).item()


def sample_budget_fraction(
    minimum: float,
    maximum: float,
    generator: Optional[Generator] = None,
) -> float:
    """
    Draw what share of the Gaussians a step is allowed to keep, 
    uniformly between 'minimum' and 'maximum'.
    """
    assert 0 < minimum <= maximum <= 1, (
        f"Need 0 < minimum <= maximum <= 1, got {minimum} and {maximum}"
    )
    return minimum + (maximum - minimum) * torch.rand(
        (1,), generator=generator
    ).item()


def fit_budget_fraction(
    fraction: float,
    minimum: float,
    maximum: float,
    num_predicted: int,
    max_gaussians: int,
) -> float:
    """
    Carry a share drawn in ['minimum', 'maximum'] onto the widest band
    of shares that the card can actually hold, given how many Gaussians
    the reconstructor predicted.
    """
    assert 0 < minimum <= maximum <= 1, (
        f"Need 0 < minimum <= maximum <= 1, got {minimum} and {maximum}"
    )
    assert num_predicted > 0 and max_gaussians > 0, (
        f"Need something to thin and room to put it in, got "
        f"{num_predicted} predicted and a ceiling of {max_gaussians}"
    )
    upper = min(maximum, max_gaussians / num_predicted)
    lower = min(minimum, upper)
    if maximum == minimum:
        return upper
    position = (fraction - minimum) / (maximum - minimum)
    return lower + (upper - lower) * position


__all__ = [
    "fit_budget_fraction",
    "sample_budget_fraction",
    "sample_num_context_views",
    "sample_view_indices",
]
