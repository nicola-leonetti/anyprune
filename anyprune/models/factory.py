"""
Building a frozen feed-forward reconstructor by name, so that a script
can name one in its configuration and every script names them alike.
"""
import torch.nn as nn

from .wrappers import FrozenAnySplat, FrozenReSplat, FrozenVolSplat, FrozenYoNoSplat


# The feed-forward reconstructors a configuration can name.
RECONSTRUCTORS = ("AnySplat", "AnySplat-voxelized", "ReSplat", "VolSplat", "YoNoSplat")

# The ReSplat checkpoint for DL3DV: the base model, trained on 8 views of
# 256x448, which is the one ReSplat's own DL3DV inference defaults to at
# that resolution
RESPLAT_CHECKPOINT = "haofeixu/resplat/resplat-base-dl3dv-256x448-view8-1934a04c.pth"

# The only VolSplat checkpoint released: trained on 6 views of
# RealEstate10K at 256x256 (there is none for DL3DV)
VOLSPLAT_CHECKPOINT = "lhmd/VolSplat/volsplat-re10k-256x256.ckpt"


def build_reconstructor(
    name: str,
    anysplat_checkpoint: str,
    yonosplat_checkpoint: str,
    quiet: bool = True,
    resplat_checkpoint: str = RESPLAT_CHECKPOINT,
    volsplat_checkpoint: str = VOLSPLAT_CHECKPOINT,
) -> nn.Module:
    """
    Build one of RECONSTRUCTORS from the checkpoint it is named with.

    ReSplat and VolSplat are posed: unlike the others they have to be
    handed the cameras of their context views (needs_cameras).
    """
    assert name in RECONSTRUCTORS, (
        f"A reconstructor has to be one of {RECONSTRUCTORS}, got {name!r}"
    )
    if name == "YoNoSplat":
        return FrozenYoNoSplat(yonosplat_checkpoint, quiet=quiet)
    if name == "ReSplat":
        return FrozenReSplat(resplat_checkpoint, quiet=quiet)
    if name == "VolSplat":
        return FrozenVolSplat(volsplat_checkpoint, quiet=quiet)
    # The same weights read two ways: the voxelized one fuses the
    # per-pixel Gaussians onto a grid inside the encoder, which is what
    # the released checkpoint configures itself for
    return FrozenAnySplat(
        anysplat_checkpoint, quiet=quiet, voxelize=name == "AnySplat-voxelized",
    )


__all__ = [
    "RECONSTRUCTORS",
    "RESPLAT_CHECKPOINT",
    "VOLSPLAT_CHECKPOINT",
    "build_reconstructor",
]
