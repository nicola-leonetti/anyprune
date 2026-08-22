"""
A model-agnostic container for the Gaussians from 3DGS.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from ..models.utils import AnySplatGaussians, build_anysplat_covariance


@dataclass
class Gaussians:
    """
    A model-agnostic representation of a 3DGS set of Gaussians.
    
    'normalized' tracks wether means and scales are normalized or not.
    By 'normalized', we mean:
    - per-scene min-max normalization of gaussians.means into [0, 1]
    - gaussians.scales shifted by the corresponding log-scale factor.
    """
    means: Tensor # (N, 3)
    covariances: Tensor # (N, 3, 3)
    harmonics: Tensor # (N, 3, d_sh), d_sh = (sh_degree + 1) ** 2
    opacities: Tensor # (N,)
    scales: Tensor # (N, 3)
    rotations: Tensor # (N, 4)

    normalized: bool

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def sh_degree(self) -> int:
        return int(self.harmonics.shape[-1] ** 0.5) - 1

    @classmethod
    def from_anysplat(cls, gaussians: AnySplatGaussians) -> "Gaussians":
        return cls(
            means=gaussians.means,
            covariances=gaussians.covariances,
            harmonics=gaussians.harmonics,
            opacities=gaussians.opacities,
            scales=gaussians.scales,
            rotations=gaussians.rotations,
            normalized=False,
        )

    @classmethod
    def from_splatformer(cls, gs: dict) -> "Gaussians":
        scales = torch.exp(gs["scales"])
        rotations = gs["quats"] / gs["quats"].norm(dim=-1, keepdim=True)
        opacities = torch.sigmoid(gs["opacities"]).squeeze(-1)

        features_dc = gs["features_dc"].unsqueeze(-1)  # (N, 3, 1)
        if "features_rest" in gs:
            features_rest = gs["features_rest"].transpose(-2, -1)  # (N, 3, d_sh - 1)
            harmonics = torch.cat([features_dc, features_rest], dim=-1)  # (N, 3, d_sh)
        else:
            harmonics = features_dc  # (N, 3, 1), sh_degree == 0

        return cls(
            means=gs["means"],
            covariances=build_anysplat_covariance(scales, rotations),
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            normalized=True,
        )
