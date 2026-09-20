"""
The per-Gaussian importance scores the pruning literature defines,
measured on a field one of this project's reconstructors has predicted.

Both scores implemented here are built out of the same quantity, the
blending weight a Gaussian carries on a ray:

    w_ij = alpha_ij * T_ij,    T_ij = prod_{k < i} (1 - alpha_kj)

which is the share of ray j's final colour that Gaussian i is
responsible for, with alpha_ij its opacity times its projected density
at that pixel and T_ij what the Gaussians in front of it left of the
ray. A Gaussian that no ray ever sees carries zero weight everywhere,
and the weights along a ray sum to the alpha that ray came back with.

RadSplat (Niemeyer et al., 2024, arXiv:2403.13806) takes the largest
weight a Gaussian ever carries, over every ray of every training view,

    h(p_i) = max_{I in images, r in I} alpha_i^r tau_i^r                (1)

and drops everything under a threshold. The maximum is the point of it:
a Gaussian that matters to one view and to nothing else is kept, where
a score that averaged or summed over the views would bury it.

Mini-Splatting (Fang and Wang, ECCV 2024, arXiv:2403.14166) sums the
weight instead,

    I_i = sum_{j=1..K} w_ij                                            (2)

over the K rays that intersect the Gaussian, and keeps a field sampled
with probability proportional to that sum rather than the top of it.
Its appendix gives a second form, the one its code reaches for on
outdoor scans,

    I_i = sum_m delta(i in Imax_m) * (sum_j w_ij_m) / S_i_m            (3)

which divides a view's weight by the area the Gaussian projects to on
it, so that a large splat covering half the frame does not outscore the
detail in front of it, and only counts the views where the Gaussian is
the largest contribution to at least one ray. That last set, Imax, is
the paper's 'intersection preserving': the Gaussians a ray actually
stops on rather than the haze it passes through. The reference
implementation applies it to (2) as well, by zeroing every Gaussian
that is never the largest weight on any ray, which is what this module
does too.

Both scores are measured over the views the reconstructor was given,
which are this setting's training views: the held-out views are what
the pruned field is scored on, and a score that had read them would be
answering a question it had already seen.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import gsplat
import torch
from torch import Tensor

from .gaussians import TILE_SIZE as _TILE_SIZE
from .gaussians import Gaussians, depth_runs, view_matrices


# The names a caller can ask for a score by, and the ones that need the
# field rendered before they can answer.
SCORES = ("uniform", "radsplat", "mini-splatting", "mini-splatting-outdoor")
MEASURED_SCORES = tuple(name for name in SCORES if name != "uniform")

# Where the rasterizer holds an alpha, copied from gsplat's kernels so
# that the weights measured here are the weights its renders are made
# of.
_MAX_ALPHA = 0.999

# How many of the depth-sorted batches of Gaussians are composited in
# one pass. Each pass holds every (pixel, Gaussian) intersection it
# finds at once, so this trades the memory of one pass against how many
# passes a view takes; a pass also gets cheaper as it goes, since the
# rasterizer stops returning intersections behind a pixel that is
# already opaque.
_BATCHES_PER_PASS = 32

# Under this much area, in square pixels, a projection is too small to
# divide a weight by: the Gaussians that land there are the ones the
# rasterizer has already blown up to its minimum footprint.
_MIN_PROJECTED_AREA = 1e-6


@dataclass
class BlendingWeights:
    """
    What one sweep over a set of views measured of every Gaussian in a
    field, as the four numbers the scores above are written out of.

    Every array is one entry per Gaussian, in the order the field holds
    them.
    """
    peak: Tensor    # (N,) the largest weight it carried on any ray
    total: Tensor   # (N,) the sum of the weights it carried
    density: Tensor # (N,) sum over views of its weight over its area
    peaked: Tensor  # (N,) bool, whether it was ever a ray's largest

    @property
    def num_gaussians(self) -> int:
        return self.peak.shape[0]


def _projected_areas(conics: Tensor) -> Tensor:
    """
    The area, in square pixels, of the one-sigma ellipse each Gaussian
    projects to, from the (C, N, 3) upper triangle [a, b, c] of the
    inverse of its 2D covariance.

    An ellipse of covariance S covers pi * sqrt(det S), and det S is one
    over the determinant a * c - b^2 of the conic gsplat hands back.
    """
    determinant = conics[..., 0] * conics[..., 2] - conics[..., 1] ** 2
    return math.pi / determinant.clamp(min=_MIN_PROJECTED_AREA).sqrt()


def _segment_starts(rays: Tensor) -> Tensor:
    """
    A mask of where each run of equal values in a sorted (M,) tensor of
    ray indices begins.
    """
    starts = torch.ones_like(rays, dtype=torch.bool)
    starts[1:] = rays[1:] != rays[:-1]
    return starts


def _weights_of_pass(
    alphas: Tensor,
    rays: Tensor,
    transmittance: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Composite one pass' intersections, sorted by ray and then by depth,
    into the weight each of them carries.

    Returns the (M,) weights, the ray each run of them ends on and the
    share of the light that run leaves behind, which is what the caller
    multiplies the running transmittance of those rays by before it asks
    for the next pass.

    The product of (1 - alpha) in front of an intersection is taken in
    logs, as a cumulative sum with the sum up to the start of its ray
    taken back off, so that the whole pass composites in a handful of
    kernels rather than one per Gaussian along a ray. The sum runs over
    every ray of the pass and so over millions of terms, where what is
    wanted of it is the handful that belong to one ray: in single
    precision the two ends of that subtraction would already have grown
    far enough apart to take the answer with them, so it is carried in
    double and only the exponentials come back down.
    """
    kept = torch.log1p(-alphas).double()
    running = torch.cumsum(kept, dim=0)
    starts = _segment_starts(rays)
    # Where each ray's own run of the cumulative sum starts from, spread
    # back over the intersections of that ray
    ray_of_run = torch.cumsum(starts.long(), dim=0) - 1
    base = (running - kept)[starts]
    # What the Gaussians in front of it on this ray left, times what
    # every earlier pass left of the ray
    in_front = torch.exp(running - kept - base[ray_of_run]).float()
    weights = alphas * in_front * transmittance[rays]

    ends = torch.zeros_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = True
    return weights, rays[ends], torch.exp(running[ends] - base).float()


@torch.no_grad()
def blending_weights(
    gaussians: Gaussians,
    poses: Tensor,
    intrinsics: Tensor,
    image_shape: Tuple[int, int],
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    batches_per_pass: int = _BATCHES_PER_PASS,
    max_intersections: Optional[int] = None,
) -> BlendingWeights:
    """
    Render the field from every one of V views and gather what each
    Gaussian contributed to it, without keeping any of the renders.

    The views are given the way Gaussians.rasterize() takes them, and
    are rendered one at a time: a sweep only ever holds the
    intersections of a single view, which is what lets it run over a
    whole prediction rather than over a field already thinned to fit.
    'max_intersections' bounds even that, the way it does in
    Gaussians.rasterize(): a view covering more tiles than it is swept
    as runs of consecutive depth, front to back, carrying the
    transmittance from one run to the next.

    The weights are the ones the rasterizer itself composites, down to
    where it clamps an alpha and where it stops walking a pixel that has
    gone opaque, so a Gaussian the renders never showed scores zero
    here.
    """
    assert gaussians.means.is_cuda, \
        "gsplat's rasterizer is CUDA only, move the Gaussians to a GPU first"
    assert batches_per_pass > 0, (
        f"A pass has to composite at least one batch, got {batches_per_pass}"
    )
    device = gaussians.device
    num_gaussians = gaussians.num_gaussians
    height, width = image_shape

    poses = poses.reshape(-1, 4, 4).to(gaussians.means)
    intrinsics = intrinsics.reshape(-1, 3, 3).to(gaussians.means)
    assert poses.shape[0] == intrinsics.shape[0], (
        f"Got {poses.shape[0]} poses but {intrinsics.shape[0]} intrinsics"
    )
    assert poses.shape[0] > 0, "There are no views to measure over"
    viewmats = view_matrices(poses)

    means = gaussians.means.float()
    # The projection wants the upper triangle of a covariance rather
    # than the whole of it, in the order gsplat's own rasterizer packs
    # it in
    rows, columns = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    covariances = gaussians.covariances.float()[..., rows, columns]
    # Read as the rasterizer reads them, which is where the weights of
    # an over-confident field stop growing
    opacities = gaussians.opacities.float().clamp(0.0, 1.0)

    measured = BlendingWeights(
        peak=torch.zeros(num_gaussians, device=device),
        total=torch.zeros(num_gaussians, device=device),
        density=torch.zeros(num_gaussians, device=device),
        peaked=torch.zeros(num_gaussians, dtype=torch.bool, device=device),
    )
    tile_width = math.ceil(width / _TILE_SIZE)
    tile_height = math.ceil(height / _TILE_SIZE)
    batch_size = _TILE_SIZE * _TILE_SIZE

    for view in range(viewmats.shape[0]):
        radii, means2d, depths, conics, _ = gsplat.fully_fused_projection(
            means, covariances, None, None,
            viewmats[view:view + 1], intrinsics[view:view + 1],
            width, height, near_plane=near_plane, far_plane=far_plane,
            packed=False, calc_compensations=False,
        )
        # The whole view at once, or as many runs of it as fit
        runs = [slice(None)]
        if max_intersections is not None:
            runs = depth_runs(
                means2d[0], radii[0], depths[0], tile_width, tile_height,
                max_intersections,
            )

        transmittance = torch.ones(1, height, width, device=device)
        view_total = torch.zeros(num_gaussians, device=device)
        # The largest weight each ray has seen so far and who carried
        # it, packed into one integer so that a single scatter answers
        # both: the bits of a non-negative float sort the way the float
        # does, so the maximum over a ray of (weight, index) is the
        # index of its maximum weight.
        ray_peak = torch.zeros(height * width, dtype=torch.int64, device=device)

        for run in runs:
            run_means2d = means2d[:, run].contiguous()
            run_conics = conics[:, run].contiguous()
            run_depths = depths[:, run]
            run_opacities = opacities[run]
            _, isect_ids, flatten_ids = gsplat.isect_tiles(
                run_means2d, radii[:, run].contiguous(), run_depths,
                _TILE_SIZE, tile_width, tile_height, packed=False, n_cameras=1,
            )
            if flatten_ids.numel() == 0:
                continue
            isect_offsets = gsplat.isect_offset_encode(
                isect_ids, 1, tile_width, tile_height
            )
            # How many batches the deepest tile of this run takes, which
            # is how far the passes below have to walk before every
            # pixel is accounted for
            edges = torch.cat([
                isect_offsets.flatten(),
                torch.tensor([flatten_ids.numel()], device=device),
            ])
            num_batches = math.ceil(
                (edges[1:] - edges[:-1]).max().item() / batch_size
            )
            view_opacities = run_opacities.unsqueeze(0).contiguous() # (1, M)

            for first in range(0, num_batches, batches_per_pass):
                gaussian_ids, pixel_ids, _ = gsplat.rasterize_to_indices_in_range(
                    first, first + batches_per_pass, transmittance,
                    run_means2d, run_conics, view_opacities, width, height,
                    _TILE_SIZE, isect_offsets, flatten_ids,
                )
                if gaussian_ids.numel() == 0:
                    break
                gaussian_ids = gaussian_ids.long()
                rays = pixel_ids.long()

                # The alpha of each intersection, as the rasterizer
                # computes it: the Gaussian's density at the centre of
                # that pixel, times its opacity
                centers = torch.stack(
                    [rays % width, rays // width], dim=-1
                ) + 0.5
                deltas = centers - run_means2d[0, gaussian_ids]
                conic = run_conics[0, gaussian_ids]
                powers = (
                    0.5 * (conic[:, 0] * deltas[:, 0] ** 2
                           + conic[:, 2] * deltas[:, 1] ** 2)
                    + conic[:, 1] * deltas[:, 0] * deltas[:, 1]
                )
                alphas = (run_opacities[gaussian_ids] * torch.exp(-powers)).clamp(
                    max=_MAX_ALPHA
                )

                # Sorted by ray, and by depth within a ray, which is the
                # order the weights composite in. The rasterizer walks a
                # tile in that order already, but it returns its hits
                # pixel by pixel rather than ray by ray.
                order = torch.argsort(run_depths[0, gaussian_ids])
                order = order[torch.argsort(rays[order], stable=True)]
                gaussian_ids, rays, alphas = (
                    gaussian_ids[order], rays[order], alphas[order]
                )
                weights, ends, left = _weights_of_pass(
                    alphas, rays, transmittance.reshape(-1)
                )
                # Back from the run's own numbering to the field's
                if isinstance(run, Tensor):
                    gaussian_ids = run[gaussian_ids]

                view_total.index_add_(0, gaussian_ids, weights)
                measured.peak.scatter_reduce_(
                    0, gaussian_ids, weights, reduce="amax"
                )
                ray_peak.scatter_reduce_(
                    0, rays,
                    (weights.contiguous().view(torch.int32).long() << 32)
                    | gaussian_ids,
                    reduce="amax",
                )
                # What this pass left of each ray it touched, for the
                # next one to start from
                flat = transmittance.reshape(-1)
                flat[ends] = flat[ends] * left
            del isect_ids, flatten_ids, isect_offsets

        # A ray whose largest weight is zero was never actually stopped
        # by anything, and the low half of its entry is a Gaussian index
        # that never carried any weight
        peaked = ray_peak[ray_peak >= (1 << 32)] & 0xFFFFFFFF
        view_peaked = torch.zeros_like(measured.peaked)
        view_peaked[peaked] = True

        measured.total += view_total
        measured.peaked |= view_peaked
        measured.density += torch.where(
            view_peaked, view_total / _projected_areas(conics)[0], 0.0
        )
    return measured


def radsplat_score(measured: BlendingWeights) -> Tensor:
    """
    RadSplat's importance, equation (1): the largest contribution a
    Gaussian ever made to a ray.
    """
    return measured.peak


def mini_splatting_score(
    measured: BlendingWeights, outdoor: bool = False
) -> Tensor:
    """
    Mini-Splatting's importance: the sum of the contributions a Gaussian
    made, equation (2), or with each view's sum divided by the area the
    Gaussian covers on it, equation (3), which is the form its code uses
    on outdoor scans.

    Either way a Gaussian that is never the largest contribution to any
    ray scores zero, which is the intersection preserving of the paper
    and the accum_area_max test of its code.
    """
    score = measured.density if outdoor else measured.total
    return torch.where(measured.peaked, score, 0.0)


def score_of(score: str, measured: BlendingWeights) -> Tensor:
    """
    One of the scores above, by name, read off a sweep that has already
    been made: every score is written out of the same four numbers, so a
    field measured once answers for all of them.
    """
    assert score in MEASURED_SCORES, (
        f"The score has to be one of {MEASURED_SCORES}: {score} is not"
    )
    if score == "radsplat":
        return radsplat_score(measured)
    return mini_splatting_score(
        measured, outdoor=score == "mini-splatting-outdoor"
    )


@torch.no_grad()
def importance_score(
    score: str,
    gaussians: Gaussians,
    poses: Tensor,
    intrinsics: Tensor,
    image_shape: Tuple[int, int],
    **kwargs,
) -> Tensor:
    """
    One of the scores above, by name, measured over the given views.

    Comes back as one non-negative number per Gaussian, larger where the
    Gaussian matters more to the views it was measured over.
    """
    return score_of(
        score, blending_weights(gaussians, poses, intrinsics, image_shape, **kwargs)
    )


__all__ = [
    "BlendingWeights",
    "MEASURED_SCORES",
    "SCORES",
    "blending_weights",
    "importance_score",
    "mini_splatting_score",
    "radsplat_score",
    "score_of",
]
