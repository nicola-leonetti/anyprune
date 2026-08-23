"""
A model-agnostic container for the Gaussians from 3DGS.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import gsplat
import torch
from torch import Tensor

from ..models.utils import (
    AnySplatGaussians, YoNoSplatGaussians, build_anysplat_covariance
)


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

    def rasterize(
        self,
        poses: Tensor,
        intrinsics: Tensor,
        image_shape: Tuple[int, int],
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        background: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Renders the Gaussians from V viewpoints and returns a tuple
        (colors, alphas) of shapes (V, 3, H, W) and (V, 1, H, W), with
        colors in [0, 1].

        Camera convention is the DL3DV/nerfstudio one:
        - 'poses' are camera-to-world matrices of shape (V, 4, 4), or of
          any shape that flattens to it, such as (V, 1, 4, 4) like in 
          DL3DV.
        - the camera axes are OpenGL/Blender, i.e. +X right, +Y up and
          +Z backwards, so the camera looks down its own -Z.
        - intrinsics are pinhole camera matrices 
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] of shape (V, 3, 3), 
            in *pixels* of the rendered image.

        Beware that AnySplat does not use this convention: its predicted
        extrinsics are camera-to-world in the OpenCV convention, and its
        intrinsics are normalized by the image size, so both have to be
        converted before they reach this method.

        The poses also have to live in the same frame as 'means': when
        `normalized` is set that is the per-scene normalized frame, not
        the world frame the original dataset's poses are given in.
        """
        assert self.means.is_cuda, \
            "gsplat's rasterizer is CUDA only, move the Gaussians to a GPU first"
        poses = poses.reshape(-1, 4, 4).to(self.means)
        intrinsics = intrinsics.reshape(-1, 3, 3).to(self.means)
        assert poses.shape[0] == intrinsics.shape[0], (
            f"Got {poses.shape[0]} poses but {intrinsics.shape[0]} intrinsics"
        )
        height, width = image_shape

        # OpenGL -> OpenCV axes, then camera-to-world -> world-to-camera
        opengl_to_opencv = torch.diag(
            torch.tensor([1.0, -1.0, -1.0, 1.0]).to(self.means)
        )
        viewmats = torch.linalg.inv(poses @ opengl_to_opencv)

        colors, alphas, _ = gsplat.rasterization(
            means=self.means,
            # We hand gsplat the covariances rather than the scales and
            # rotations: the quaternion layout of `rotations` is
            # whichever one the source model used (AnySplat's is xyzw,
            # gsplat expects wxyz), while the covariances mean the same
            # thing whoever built them.
            quats=None,
            scales=None,
            opacities=self.opacities,
            colors=self.harmonics.transpose(-2, -1).contiguous(), # (N, d_sh, 3)
            viewmats=viewmats,
            Ks=intrinsics,
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=self.sh_degree,
            backgrounds=background,
            covars=self.covariances,
        )
        colors = colors.clamp(0.0, 1.0).permute(0, 3, 1, 2) # (V, 3, H, W)
        alphas = alphas.permute(0, 3, 1, 2)                 # (V, 1, H, W)
        return colors, alphas

    @classmethod
    def from_anysplat(cls, gaussians: AnySplatGaussians) -> "Gaussians":
        assert gaussians.means.shape[0] == 1, (
            f"Expected a single scene from AnySplat, got a batch of {gaussians.means.shape[0]}"
        )
        return cls(
            means=gaussians.means[0],
            covariances=gaussians.covariances[0],
            harmonics=gaussians.harmonics[0],
            opacities=gaussians.opacities[0],
            scales=gaussians.scales[0],
            rotations=gaussians.rotations[0],
            normalized=False,
        )

    @classmethod
    def from_yonosplat(cls, gaussians: YoNoSplatGaussians) -> "Gaussians":
        assert gaussians.means.shape[0] == 1, (
            f"Expected a single scene from YoNoSplat, got a batch of {gaussians.means.shape[0]}"
        )
        return cls(
            means=gaussians.means[0],
            covariances=gaussians.covariances[0],
            harmonics=gaussians.harmonics[0],
            opacities=gaussians.opacities[0],
            scales=gaussians.scales[0],
            rotations=gaussians.rotations[0],
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
