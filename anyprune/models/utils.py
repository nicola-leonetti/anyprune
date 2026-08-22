"""
Helpers borrowed from external model codebases (e.g. AnySplat),
to be used elsewhere in the project.
"""
from src.model.types import Gaussians as AnySplatGaussians
from src.model.encoder.common.gaussians import build_covariance as build_anysplat_covariance


__all__ = [
    "AnySplatGaussians",
    "build_anysplat_covariance",
]