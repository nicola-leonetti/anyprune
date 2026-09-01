"""
A model-agnostic container for the Gaussians from 3DGS.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import gsplat
import torch
from torch import Generator, Tensor

from ..models.utils import AnySplatGaussians, YoNoSplatGaussians


@dataclass
class Gaussians:
    """
    A model-agnostic representation of a 3DGS set of Gaussians.


    Parameters are not normalized and are expressed in the same world
    frame as the cameras.
    """
    means: Tensor # (N, 3)
    covariances: Tensor # (N, 3, 3)
    harmonics: Tensor # (N, 3, d_sh), d_sh = (sh_degree + 1) ** 2
    opacities: Tensor # (N,)
    scales: Tensor # (N, 3)
    rotations: Tensor # (N, 4)

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
        views_per_pass: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Renders the Gaussians from V poses and returns a tuple
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

        (note that AnySplat does not use this convention: its predicted
        extrinsics are camera-to-world in the OpenCV convention, and its
        intrinsics are normalized by the image size, so both have to be
        converted before they reach this method).

        The poses also have to live in the same frame as 'means'.

        'views_per_pass' renders the views in groups of that size rather
        than all at once to avoid allocating too much memory at once and
        does not influence the final rendered result.
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

        num_views = viewmats.shape[0]
        assert num_views > 0, "There are no views to render from"
        assert background is None or background.shape[0] == num_views, (
            "gsplat wants one background per view, got "
            f"{background.shape[0]} for {num_views} views"
        )
        assert views_per_pass is None or views_per_pass > 0, (
            f"Cannot render {views_per_pass} views at a time"
        )
        # A camera is rasterized independently of every other one, so
        # the groups only decide what is in flight at once
        harmonics = self.harmonics.transpose(-2, -1).contiguous() # (N, d_sh, 3)
        rendered = []
        for first in range(0, num_views, views_per_pass or num_views):
            group = slice(first, first + (views_per_pass or num_views))
            colors, alphas, _ = gsplat.rasterization(
                means=self.means,
                # We hand gsplat the covariances rather than the scales
                # and rotations because covariances are 
                # model-independent
                quats=None,
                scales=None,
                opacities=self.opacities,
                colors=harmonics,
                viewmats=viewmats[group],
                Ks=intrinsics[group],
                width=width,
                height=height,
                near_plane=near_plane,
                far_plane=far_plane,
                sh_degree=self.sh_degree,
                backgrounds=None if background is None else background[group],
                covars=self.covariances,
            )
            rendered.append((colors, alphas))
        colors = torch.cat([colors for colors, _ in rendered], dim=0)
        alphas = torch.cat([alphas for _, alphas in rendered], dim=0)
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
        )

    def subsample(
        self, num_gaussians: int, generator: Optional[Generator] = None
    ) -> "Gaussians":
        """
        Draw `num_gaussians` of the Gaussians uniformly at random,
        returning the set unchanged if it is already that small.

        Optionally accepts a PyTorch Generator to leave the GPU RNG
        untouched.
        """
        if self.num_gaussians <= num_gaussians: return self
        # Drawn on the CPU so that the caller's generator, which seeds
        # the view sampling too, does not have to live on the GPU
        kept = torch.randperm(self.num_gaussians, generator=generator)[:num_gaussians]
        kept = kept.to(self.device)
        return replace(
            self,
            means=self.means[kept],
            covariances=self.covariances[kept],
            harmonics=self.harmonics[kept],
            opacities=self.opacities[kept],
            scales=self.scales[kept],
            rotations=self.rotations[kept],
        )
