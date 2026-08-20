"""
This module contains wrappers around external modules (e.g. Anysplat) to
provide a simplified interface for use in the rest of the project.
"""
from .frozen_anysplat import FrozenAnySplat

__all__ = [
    "FrozenAnySplat",
]