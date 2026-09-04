"""
Building a frozen feed-forward reconstructor by name, so that a script
can name one in its configuration and every script names them alike.
"""
import torch.nn as nn

from .wrappers import FrozenAnySplat, FrozenYoNoSplat


# The feed-forward reconstructors a configuration can name.
RECONSTRUCTORS = ("AnySplat", "AnySplat-voxelized", "YoNoSplat")


def build_reconstructor(
    name: str,
    anysplat_checkpoint: str,
    yonosplat_checkpoint: str,
    quiet: bool = True,
) -> nn.Module:
    """Build one of RECONSTRUCTORS from the checkpoint it is named with."""
    assert name in RECONSTRUCTORS, (
        f"A reconstructor has to be one of {RECONSTRUCTORS}, got {name!r}"
    )
    if name == "YoNoSplat":
        return FrozenYoNoSplat(yonosplat_checkpoint, quiet=quiet)
    # The same weights read two ways: the voxelized one fuses the
    # per-pixel Gaussians onto a grid inside the encoder, which is what
    # the released checkpoint configures itself for
    return FrozenAnySplat(
        anysplat_checkpoint, quiet=quiet, voxelize=name == "AnySplat-voxelized",
    )


__all__ = [
    "RECONSTRUCTORS",
    "build_reconstructor",
]
