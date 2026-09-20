"""
A model-agnostic container for the Gaussians from 3DGS.
"""
import math
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import gsplat
import torch
from torch import Generator, Tensor

from ..models.utils import AnySplatGaussians, YoNoSplatGaussians


# The tile an image is cut into, which is the one gsplat is built around
# and the only size it is tested at.
TILE_SIZE = 16

# The upper triangle of a covariance, in the order gsplat packs it in.
_TRIANGLE = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])


def _tile_intersections(
    means2d: Tensor, radii: Tensor, tile_width: int, tile_height: int
) -> Tensor:
    """
    How many tiles each of N projected Gaussians covers, counted the way
    gsplat.isect_tiles() counts them, from their (N, 2) centres and (N,)
    radii in pixels. A culled Gaussian, of radius 0, covers none.
    """
    radii = radii.unsqueeze(-1).float()
    bounds = torch.tensor([tile_width, tile_height], device=means2d.device)
    low = ((means2d - radii) / TILE_SIZE).floor().clamp(min=0).minimum(bounds)
    high = ((means2d + radii) / TILE_SIZE).ceil().clamp(min=0).minimum(bounds)
    return torch.where(radii[:, 0] > 0, (high - low).prod(dim=-1), 0.0).long()


def depth_runs(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_width: int,
    tile_height: int,
    max_intersections: int,
) -> List[Tensor]:
    """
    Split the Gaussians visible in one view into runs of consecutive
    depth, front to back, each covering at most 'max_intersections'
    tiles, as lists of indices into the field.

    Alpha blending composites run by run exactly as it does Gaussian by
    Gaussian, so a view too large to rasterize at once can be rendered
    as its runs and put together with the transmittance each one leaves.
    The projection is given as gsplat.fully_fused_projection() returns
    it for a single camera.
    """
    assert max_intersections > 0, (
        f"A run has to hold at least one intersection, got {max_intersections}"
    )
    counts = _tile_intersections(means2d, radii, tile_width, tile_height)
    visible = torch.nonzero(counts > 0).squeeze(-1)
    order = visible[torch.argsort(depths[visible])]
    cumulative = torch.cumsum(counts[order], dim=0)
    runs, first = [], 0
    while first < order.shape[0]:
        limit = (cumulative[first - 1] if first > 0 else 0) + max_intersections
        last = int(torch.searchsorted(cumulative, limit, right=True).item())
        # A single Gaussian over the budget is a run of its own
        last = max(last, first + 1)
        runs.append(order[first:last])
        first = last
    return runs


def view_matrices(poses: Tensor) -> Tensor:
    """
    The world-to-camera matrices gsplat rasterizes with, for poses given
    in the DL3DV/nerfstudio convention: camera-to-world, with the camera
    looking down its own -Z.

    'poses' is of shape (V, 4, 4) or of any shape that flattens to it,
    and carries the dtype and device the result comes back on.
    """
    poses = poses.reshape(-1, 4, 4)
    # OpenGL -> OpenCV axes, then camera-to-world -> world-to-camera
    opengl_to_opencv = torch.diag(
        torch.tensor([1.0, -1.0, -1.0, 1.0]).to(poses)
    )
    return torch.linalg.inv(poses @ opengl_to_opencv)


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

    def __getitem__(self, index) -> "Gaussians":
        """
        The Gaussians at 'index', which is anything a tensor takes: a
        slice, a mask, or indices, kept in the order they are given in.
        """
        return replace(
            self,
            means=self.means[index],
            covariances=self.covariances[index],
            harmonics=self.harmonics[index],
            opacities=self.opacities[index],
            scales=self.scales[index],
            rotations=self.rotations[index],
        )

    def rasterize(
        self,
        poses: Tensor,
        intrinsics: Tensor,
        image_shape: Tuple[int, int],
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        background: Optional[Tensor] = None,
        views_per_pass: Optional[int] = None,
        max_intersections: Optional[int] = None,
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

        'max_intersections' bounds the memory of a single view instead:
        the views are then rendered one at a time, and one whose
        Gaussians cover more tiles than that is rendered as runs of
        consecutive depth composited front to back, which is the same
        image the rasterizer would make of it all at once. A field of
        millions of Gaussians, some of them close to a camera, covers
        hundreds of millions of tiles in a view, which the rasterizer
        cannot sort in memory in one go.
        """
        assert self.means.is_cuda, \
            "gsplat's rasterizer is CUDA only, move the Gaussians to a GPU first"
        poses = poses.reshape(-1, 4, 4).to(self.means)
        intrinsics = intrinsics.reshape(-1, 3, 3).to(self.means)
        assert poses.shape[0] == intrinsics.shape[0], (
            f"Got {poses.shape[0]} poses but {intrinsics.shape[0]} intrinsics"
        )
        height, width = image_shape

        viewmats = view_matrices(poses)

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

        def rasterize(group: slice, kept=slice(None), shaded=True) -> Tuple[Tensor, Tensor]:
            colors, alphas, _ = gsplat.rasterization(
                means=self.means[kept],
                # We hand gsplat the covariances rather than the scales
                # and rotations because covariances are 
                # model-independent
                quats=None,
                scales=None,
                opacities=self.opacities[kept],
                colors=harmonics[kept],
                viewmats=viewmats[group],
                Ks=intrinsics[group],
                width=width,
                height=height,
                near_plane=near_plane,
                far_plane=far_plane,
                sh_degree=self.sh_degree,
                backgrounds=(
                    None if background is None or not shaded else background[group]
                ),
                covars=self.covariances[kept],
            )
            return colors, alphas

        rendered = []
        if max_intersections is None:
            for first in range(0, num_views, views_per_pass or num_views):
                rendered.append(rasterize(
                    slice(first, first + (views_per_pass or num_views))
                ))
        else:
            for view in range(num_views):
                group = slice(view, view + 1)
                radii, means2d, depths, _, _ = gsplat.fully_fused_projection(
                    self.means.float(), self.covariances.float()[..., _TRIANGLE[0], _TRIANGLE[1]],
                    None, None, viewmats[group], intrinsics[group], width, height,
                    near_plane=near_plane, far_plane=far_plane,
                    packed=False, calc_compensations=False,
                )
                runs = depth_runs(
                    means2d[0], radii[0], depths[0],
                    math.ceil(width / TILE_SIZE), math.ceil(height / TILE_SIZE),
                    max_intersections,
                )
                del radii, means2d, depths
                if len(runs) <= 1:
                    rendered.append(rasterize(group))
                    continue
                # Each run is drawn on its own and then composited behind
                # what the runs in front of it left of the light. The
                # background, which gsplat would put behind each run,
                # goes behind all of them instead.
                colors = torch.zeros(1, height, width, 3, device=self.device)
                alphas = torch.zeros(1, height, width, 1, device=self.device)
                for run in runs:
                    run_colors, run_alphas = rasterize(group, run, shaded=False)
                    remaining = 1.0 - alphas
                    colors += remaining * run_colors
                    alphas += remaining * run_alphas
                if background is not None:
                    colors += (1.0 - alphas) * background[group].view(1, 1, 1, 3)
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

    def compensate(
        self, fraction: float, exponent: float = 1.0
    ) -> "Gaussians":
        """
        Put back the optical depth that thinning a field to 'fraction'
        of itself takes out of it, by raising every opacity so that a
        ray is stopped about as much as the whole field stopped it.

        A ray keeps a share (1 - a) of its light at each Gaussian it
        meets, and meets 'fraction' as many of them once the field is
        thinned, so the opacity that leaves that product where it was is

            a' = 1 - (1 - a) ** (exponent / fraction)

        'exponent' at 1 is what that identity implies; above 1 it
        corrects harder than the independence the identity assumes.

        The scales are left alone: inflating them restores the same
        missing depth a second time.

        A field that was not thinned comes back unchanged.
        """
        assert 0.0 < fraction <= 1.0, (
            f"A field can only be thinned to a share of itself, got {fraction}"
        )
        assert exponent > 0.0, f"Need a positive exponent, got {exponent}"
        if fraction == 1.0 and exponent == 1.0:
            return self
        opacities = self.opacities.clamp(0.0, 1.0)
        return replace(
            self, opacities=1.0 - (1.0 - opacities) ** (exponent / fraction)
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
        return self[kept.to(self.device)]
