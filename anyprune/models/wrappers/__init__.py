"""
This module contains wrappers around external modules (e.g. Anysplat) to
provide a simplified interface for use in the rest of the project.
"""
from .frozen_anysplat import FrozenAnySplat
from .frozen_resplat import FrozenReSplat
from .frozen_volsplat import FrozenVolSplat
from .frozen_yonosplat import FrozenYoNoSplat
from .splatformer import SplatFormer

__all__ = [
    "FrozenAnySplat",
    "FrozenReSplat",
    "FrozenVolSplat",
    "FrozenYoNoSplat",
    "SplatFormer",
]