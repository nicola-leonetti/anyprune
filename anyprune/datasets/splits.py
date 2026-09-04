"""
Dividing the scenes of a dataset between the splits a run trains,
validates and reports on.
"""
from typing import Dict, List, Mapping

import torch
from torch import Generator


def split_scenes(
    num_scenes: int,
    fractions: Mapping[str, float],
    seed: int = 0,
) -> Dict[str, List[int]]:
    """
    Divide 'num_scenes' scenes between the splits, in the proportions
    'fractions' names and always the same way.

    The order comes off 'seed' alone, so a run that only evaluates lands
    on the same division as the run that trained: the scenes reported on
    are the ones training never saw, as long as both are given the same
    seed and the same proportions.

    Rounding can leave a scene over, which goes to the first split.
    """
    assert fractions, "Need at least one split to divide the scenes between"
    order = torch.randperm(
        num_scenes, generator=Generator().manual_seed(seed)
    ).tolist()
    splits, first = {}, 0
    for name, fraction in fractions.items():
        last = first + round(fraction * num_scenes)
        splits[name] = order[first:last]
        first = last
    splits[next(iter(fractions))].extend(order[first:])
    return splits


__all__ = [
    "split_scenes",
]
