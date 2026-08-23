"""
Models, both from this project and other projects, to be used in the
rest of the codebase.

External codebases are only ever reached through the wrappers below,
which give them a common, simplified interface.
"""
from .wrappers import FrozenAnySplat, FrozenYoNoSplat


__all__ = [
    "FrozenAnySplat",
    "FrozenYoNoSplat",
]