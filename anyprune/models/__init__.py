"""
Models, both from this project and other projects, to be used in the 
rest of the codebase.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "external" / "AnySplat"))
from src.model.model import AnySplat


__all__ = [
    "AnySplat",
]
