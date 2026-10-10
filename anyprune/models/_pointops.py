"""
A stand-in, in plain PyTorch, for the two functions of the 'pointops'
CUDA extension that ReSplat calls: knn_query() and
knn_query_and_group().

ReSplat ships pointops as sources to compile against the CUDA toolkit
PyTorch was built with, which this machine does not have. All its point
transformers ask of it is each point's k nearest neighbours and their
features gathered, which pytorch3d's exact CUDA query does on the point
counts ReSplat makes (12,544 per view at 448x448). A torch.cdist query
stood here until 2026-10-06: under TF32 matmuls, which every script turns
on, it misplaced distances by up to 0.33 and so returned the wrong
neighbours, and at 8 views it took 4 of the 9 s of a ReSplat forward.
"""
from types import ModuleType
from typing import Optional, Tuple

import torch
from pytorch3d.ops import knn_points
from torch import Tensor


def _segments(offset: Tensor):
    """The [start, end) of each cloud of a batch packed by 'offset', pointops' cumulative counts."""
    ends = offset.tolist()
    return zip([0] + ends[:-1], ends)


@torch.no_grad()
def knn_query(
    nsample: int,
    xyz: Tensor,
    offset: Tensor,
    new_xyz: Optional[Tensor] = None,
    new_offset: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    For every point of 'new_xyz' (M, 3), the indices into 'xyz' (N, 3)
    of its 'nsample' nearest points of the same cloud, nearest first,
    and their distances, both (M, nsample), as pointops.knn_query.
    """
    if new_xyz is None or new_offset is None:
        new_xyz, new_offset = xyz, offset
    idx = torch.empty(new_xyz.shape[0], nsample, dtype=torch.int32, device=xyz.device)
    dist = torch.empty(new_xyz.shape[0], nsample, dtype=torch.float32, device=xyz.device)
    for (start, end), (new_start, new_end) in zip(_segments(offset), _segments(new_offset)):
        assert end - start >= nsample, (
            f"A cloud of {end - start} points has no {nsample} nearest neighbours"
        )
        squared, nearest, _ = knn_points(
            new_xyz[None, new_start:new_end].float(), xyz[None, start:end].float(),
            K=nsample, return_sorted=True,
        )
        idx[new_start:new_end] = (nearest[0] + start).int()
        dist[new_start:new_end] = squared[0].sqrt()
    return idx, dist


def knn_query_and_group(
    feat: Tensor,
    xyz: Tensor,
    offset: Optional[Tensor] = None,
    new_xyz: Optional[Tensor] = None,
    new_offset: Optional[Tensor] = None,
    idx: Optional[Tensor] = None,
    nsample: Optional[int] = None,
    with_xyz: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    The (M, nsample, C) features 'feat' (N, C) of every query point's
    nearest neighbours, measured with knn_query() unless 'idx' already
    holds them, and those indices, as pointops.knn_query_and_group.
    """
    assert not with_xyz, "ReSplat never groups the coordinates along with the features"
    if idx is None:
        assert nsample is not None
        idx, _ = knn_query(nsample, xyz, offset, new_xyz, new_offset)
    return feat[idx.long()], idx


def as_module() -> ModuleType:
    """These functions as a module, to stand where ReSplat imports 'pointops' from."""
    module = ModuleType("pointops")
    module.knn_query = knn_query
    module.knn_query_and_group = knn_query_and_group
    return module


__all__ = [
    "as_module",
    "knn_query",
    "knn_query_and_group",
]
