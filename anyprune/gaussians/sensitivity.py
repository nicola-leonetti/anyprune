"""
The two sensitivity scores of Hanson et al., measured on a predicted
field the way their per-scene code measures them on an optimized one:
from the derivative of every rendered pixel with respect to every
Gaussian, pixel by pixel, over the training views.

PUP 3D-GS (CVPR 2025, arXiv:2406.10219, anyprune/gaussians/external/
PUP-3DGS) scores a Gaussian by the log determinant of the Fisher
information of the L2 reconstruction error with respect to its mean
and its log scales,

    F_i = sum_{I, p, c} g_ipc g_ipc^T,   g_ipc = d I_pc / d (mu_i, log s_i)   (6 x 6)
    U_i = log det F_i

summed over every training image I, pixel p and colour channel c
(fisher_pool_xyz_scaling.py at pool resolution 1), and prunes the
lowest. Speedy-Splat (CVPR 2025, arXiv:2412.00578, anyprune/gaussians/
external/Speedy-Splat) keeps the idea and drops the Jacobian: its
rasterizer's backward accumulates, per pixel, the square of the
derivative of the image sum with respect to the Gaussian's kernel
value G_ip (the density before the opacity multiplies it),

    S_i = sum_{I, p} (o_i * d(sum_c I_pc) / d alpha_ip)^2

(backward.cu: dL_dG = con_o.w * dL_dalpha, dL_dG2 += dL_dG^2). Both are
per-pixel sums of squares, which one backward of the rasterizer does
not give, so both are taken here inside the front-to-back sweep of
anyprune.gaussians.importance, from what the rasterizer's own backward
is made of. Along a ray composited front to back,

    d I_pc / d alpha_ip = T_ip * c_ic - (I_pc - P_ipc) / (1 - alpha_ip)

with T the transmittance that reached the Gaussian, c its colour on
that view, I the rendered pixel and P the colour accumulated on the ray
up to and including the Gaussian: what the Gaussian adds, less what
raising its alpha takes from everything behind it, which is
T * (c - accum_rec) in the reference kernels. That derivative then
goes through the kernel, alpha = o * exp(-power), to the projected mean
and conic, and through the projection to the mean and log scales for
the Fisher; the mean also moves the colour, through the viewing
direction of the harmonics, and that path is added the way the
rasterizer's backward carries it. The renders composite onto black, as this project's do.

Both scores are measured over the views the reconstructor was given,
like the blending-weight scores.
"""
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import gsplat
import torch
from torch import Tensor

from .gaussians import Gaussians, view_matrices
from .importance import _BATCHES_PER_PASS, sweep

# The sweep's intersections are worked in slices of at most this many,
# since every one of them carries a 6-vector and 21 products here
_INTERSECTIONS_PER_SLICE = 2_000_000

# Row and column of each of the 21 entries of the upper triangle of a
# 6 x 6 symmetric matrix, in the order the Fisher is accumulated in
_ROWS, _COLUMNS = zip(*[(r, c) for r in range(6) for c in range(r, 6)])
_ROWS, _COLUMNS = list(_ROWS), list(_COLUMNS)


@dataclass
class Sensitivities:
    """
    What one sweep over a set of views measured of every Gaussian, one
    entry per Gaussian in the order the field holds them.
    """
    # (N, 21) the upper triangle of PUP's Fisher, mean then log scales,
    # or None when only Speedy-Splat's was asked for
    fisher: Optional[Tensor]
    # (N,) Speedy-Splat's sum of squared kernel derivatives
    speedy: Tensor

    @property
    def num_gaussians(self) -> int:
        return self.speedy.shape[0]


def _quaternion_to_matrix(xyzw: Tensor) -> Tensor:
    """(N, 4) unit quaternions, x y z w, to (N, 3, 3) rotations."""
    x, y, z, w = xyzw.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def projection_jacobian(
    gaussians: Gaussians,
    viewmat: Tensor,
    intrinsics: Tensor,
    image_shape: Tuple[int, int],
    near_plane: float,
    far_plane: float,
) -> Tensor:
    """
    (N, 5, 6): how a view's projected mean (2) and conic (3) of every
    Gaussian move with its mean (3) and log scales (3), through the
    rasterizer's own projection. Each Gaussian projects on its own, so
    the derivative of the sum over the field of one output is that
    output's derivative per Gaussian, and five backward passes give the
    whole Jacobian.
    """
    height, width = image_shape
    means = gaussians.means.float().detach().requires_grad_(True)
    log_scales = gaussians.scales.float().clamp_min(1e-12).log().detach().requires_grad_(True)
    rotation = _quaternion_to_matrix(
        torch.nn.functional.normalize(gaussians.rotations.float(), dim=-1)
    )
    with torch.enable_grad():
        scaled = rotation * torch.exp(log_scales).unsqueeze(-2)         # R S
        covariances = scaled @ scaled.transpose(-1, -2)                  # R S S^T R^T
        rows, columns = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        _, means2d, _, conics, _ = gsplat.fully_fused_projection(
            means, covariances[..., rows, columns], None, None,
            viewmat.reshape(1, 4, 4), intrinsics.reshape(1, 3, 3), width, height,
            near_plane=near_plane, far_plane=far_plane, packed=False,
            calc_compensations=False,
        )
        outputs = torch.cat([means2d[0], conics[0]], dim=-1)             # (N, 5)
        jacobian = torch.zeros(
            gaussians.num_gaussians, 5, 6, device=means.device, dtype=torch.float32
        )
        for k in range(5):
            grads = torch.autograd.grad(
                outputs[:, k].sum(), (means, log_scales), retain_graph=k < 4,
                allow_unused=True,
            )
            jacobian[:, k, :3] = 0.0 if grads[0] is None else grads[0]
            jacobian[:, k, 3:] = 0.0 if grads[1] is None else grads[1]
    return jacobian


def _view_colours(
    gaussians: Gaussians, camera_position: Tensor, jacobian: bool = True
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    (N, 3) the colour each Gaussian shows a camera at this position, as
    the rasterizer shades it: its harmonics evaluated along the
    direction from the camera, offset and clamped at zero; and (N, 3, 3)
    how that colour moves with the Gaussian's mean, channel by axis,
    through the direction, which the rasterizer's backward carries into
    the mean too (None when not asked for).
    """
    coefficients = gaussians.harmonics.float().transpose(-2, -1).contiguous()  # (N, K, 3)
    if not jacobian:
        directions = gaussians.means.float() - camera_position.reshape(1, 3)
        return (gsplat.spherical_harmonics(gaussians.sh_degree, directions, coefficients) + 0.5).clamp_min(0.0), None
    means = gaussians.means.float().detach().requires_grad_(True)
    with torch.enable_grad():
        directions = means - camera_position.reshape(1, 3)
        colours = (gsplat.spherical_harmonics(gaussians.sh_degree, directions, coefficients) + 0.5).clamp_min(0.0)
        jacobian = torch.stack([
            torch.autograd.grad(colours[:, c].sum(), means, retain_graph=c < 2)[0]
            for c in range(3)
        ], dim=1)                                                                 # (N, 3, 3)
    return colours.detach(), jacobian


@dataclass
class Derivatives:
    """
    One slice of a sweep's intersections with their derivatives, every
    array (M,) or (M, k): the view, the Gaussian and ray of each, and
    the derivative of the ray's three channels with respect to the
    intersection's alpha (M, 3), of its alpha with respect to the
    Gaussian's mean and log scales (M, 6), and of the channels with
    respect to the mean through the colour (M, 3, 3): the blending
    weight times the colour's Jacobian, which adds to the first three
    of the six.
    """
    view: int
    gaussian_ids: Tensor
    rays: Tensor
    alphas: Tensor
    d_alpha: Tensor
    d_params: Optional[Tensor]
    d_colour: Optional[Tensor]


@torch.no_grad()
def derivatives(
    gaussians: Gaussians,
    poses: Tensor,
    intrinsics: Tensor,
    image_shape: Tuple[int, int],
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    batches_per_pass: int = _BATCHES_PER_PASS,
    max_intersections: Optional[int] = None,
    fisher: bool = True,
) -> Iterator[Derivatives]:
    """
    The per-pixel derivatives of the module docstring, over V views
    given the way Gaussians.rasterize() takes them, a slice of
    intersections at a time.

    With 'fisher' off only the derivative with respect to the alpha is
    taken, which is all Speedy-Splat's score reads, and d_params and
    d_colour come back None: it saves the projection's Jacobian (five
    backward passes a view) and the products per intersection, which
    are most of the cost of a sweep.
    """
    device = gaussians.device
    height, width = image_shape
    poses = poses.reshape(-1, 4, 4).to(gaussians.means)
    intrinsics = intrinsics.reshape(-1, 3, 3).to(gaussians.means)
    viewmats = view_matrices(poses)

    current = None
    colours = colour_jacobian = image = jacobian = prefix = means2d = conics = None

    for it in sweep(
        gaussians, poses, intrinsics, image_shape, near_plane, far_plane,
        batches_per_pass, max_intersections,
    ):
        if it.view != current:
            current = it.view
            colours, colour_jacobian = _view_colours(gaussians, poses[current, :3, 3], jacobian=fisher)
            # The rendered view, flattened to (H * W, 3), which is what
            # the accumulated colour of a ray is taken back from. The
            # rasterizer clamps it to [0, 1] on the way out, which the
            # reference kernels do not, so a ray that overshoots one is
            # read as one here.
            image, _ = gaussians.rasterize(
                poses[current], intrinsics[current], image_shape,
                near_plane=near_plane, far_plane=far_plane,
                max_intersections=max_intersections,
            )
            image = image[0].float().permute(1, 2, 0).reshape(-1, 3)
            jacobian = None if not fisher else projection_jacobian(
                gaussians, viewmats[current], intrinsics[current], image_shape,
                near_plane, far_plane,
            )
            # The colour accumulated on each ray by the passes so far
            prefix = torch.zeros(height * width, 3, device=device)
            means2d, conics = it.means2d, it.conics

        # In slices, each carrying its last ray's colour into the next
        # through 'prefix', the way a pass carries into the next
        for first in range(0, it.rays.numel(), _INTERSECTIONS_PER_SLICE):
            part = slice(first, first + _INTERSECTIONS_PER_SLICE)
            ids, rays, alphas = it.gaussian_ids[part], it.rays[part], it.alphas[part]
            reached, weights = it.transmittance[part], it.weights[part]

            # What each intersection adds to its ray, and the ray's
            # colour up to and including it: a running sum along the
            # slice with the sum up to the start of the ray taken off,
            # plus what earlier slices and passes accumulated on it
            added = (weights.unsqueeze(-1) * colours[ids]).double()
            # Scanned channel by channel along the intersections: a
            # scan down the outer dimension of an (M, 3) tensor is
            # thirty times slower than one along the inner
            running = torch.cumsum(added.T.contiguous(), dim=1).T
            starts = torch.ones_like(rays, dtype=torch.bool)
            starts[1:] = rays[1:] != rays[:-1]
            ray_of_run = torch.cumsum(starts.long(), dim=0) - 1
            base = (running - added)[starts]
            accumulated = (running - base[ray_of_run]).float() + prefix[rays]
            ends = torch.zeros_like(starts)
            ends[:-1] = starts[1:]
            ends[-1] = True
            prefix[rays[ends]] = accumulated[ends]
            del added, running, base, ray_of_run

            # d I_pc / d alpha_ip for the three channels: what the
            # Gaussian adds, less what raising its alpha takes from
            # everything behind it
            behind = (image[rays] - accumulated) / (1.0 - alphas).unsqueeze(-1)
            d_alpha = reached.unsqueeze(-1) * colours[ids] - behind                  # (M, 3)
            del accumulated, behind
            if not fisher:
                yield Derivatives(current, ids, rays, alphas, d_alpha, None, None)
                continue

            # Through the kernel: alpha = o exp(-power) with
            # power = (a dx^2 + c dy^2) / 2 + b dx dy, dx dy the pixel
            # centre less the projected mean
            centres = torch.stack([rays % width, rays // width], dim=-1).float() + 0.5
            conic = conics[ids]
            deltas = centres - means2d[ids]
            dx, dy = deltas[:, 0], deltas[:, 1]
            a, b, c = conic[:, 0], conic[:, 1], conic[:, 2]
            d_kernel = alphas.unsqueeze(-1) * torch.stack([
                a * dx + b * dy, c * dy + b * dx,            # d alpha / d mean2d
                -0.5 * dx * dx, -dx * dy, -0.5 * dy * dy,    # d alpha / d conic
            ], dim=-1)                                                       # (M, 5)
            # And through the projection to the mean and log scales
            # As a product and a sum rather than a batched matmul, which
            # dispatches millions of 1 x 5 by 5 x 6 products to a GEMM
            # kernel a hundred times too big for them
            d_params = (d_kernel.unsqueeze(-1) * jacobian[ids]).sum(dim=1)          # (M, 6)
            del centres, conic, deltas, d_kernel
            d_colour = weights.reshape(-1, 1, 1) * colour_jacobian[ids]              # (M, 3, 3)
            yield Derivatives(current, ids, rays, alphas, d_alpha, d_params, d_colour)


@torch.no_grad()
def sensitivity_scores(
    gaussians: Gaussians,
    poses: Tensor,
    intrinsics: Tensor,
    image_shape: Tuple[int, int],
    fisher: bool = True,
    **kwargs,
) -> Sensitivities:
    """
    Both sensitivities of the module docstring, measured over V views
    given the way Gaussians.rasterize() takes them, in one sweep
    (derivatives() takes the keyword arguments). With 'fisher' off only
    Speedy-Splat's is measured, at about the cost of a RadSplat sweep,
    and the Fisher comes back None.
    """
    opacities = gaussians.opacities.float().clamp(0.0, 1.0)
    measured = Sensitivities(
        fisher=None if not fisher else torch.zeros(gaussians.num_gaussians, 21, device=gaussians.device),
        speedy=torch.zeros(gaussians.num_gaussians, device=gaussians.device),
    )
    for it in derivatives(gaussians, poses, intrinsics, image_shape, fisher=fisher, **kwargs):
        measured.speedy.index_add_(
            0, it.gaussian_ids, (opacities[it.gaussian_ids] * it.d_alpha.sum(dim=-1)) ** 2
        )
        if not fisher:
            continue
        # sum_c g_c g_c^T over the channels, the 21 entries of its upper
        # triangle, with g_c the channel's derivative with respect to
        # the mean and log scales: through the alpha, and through the
        # colour for the mean
        for c in range(3):
            g = it.d_alpha[:, c].unsqueeze(-1) * it.d_params                         # (M, 6)
            g[:, :3] += it.d_colour[:, c]
            measured.fisher.index_add_(0, it.gaussian_ids, g[:, _ROWS] * g[:, _COLUMNS])
    return measured


def pup_score(measured: Sensitivities) -> Tensor:
    """
    PUP 3D-GS's sensitivity: the log determinant of the Fisher, as the
    sum of the logs of its singular values (prune_finetune.py), which
    for the symmetric positive semi-definite Fisher are its eigenvalues.
    A Gaussian no view ever showed has a Fisher of zeros; its
    eigenvalues are floored so that it scores lowest rather than -inf.
    """
    assert measured.fisher is not None, "The Fisher was not measured (sensitivity_scores(fisher=False))"
    score = torch.empty(measured.num_gaussians, device=measured.fisher.device)
    # In slices: the batched eigensolver's workspace is many times the
    # matrices' own size
    for first in range(0, measured.num_gaussians, 65_536):
        part = measured.fisher[first:first + 65_536].double()
        fisher = torch.zeros(part.shape[0], 6, 6, device=part.device, dtype=torch.float64)
        fisher[:, _ROWS, _COLUMNS] = part
        fisher[:, _COLUMNS, _ROWS] = part
        eigenvalues = torch.linalg.eigvalsh(fisher)
        score[first:first + 65_536] = eigenvalues.clamp_min(1e-30).log().sum(dim=-1).float()
    return score


def speedy_splat_score(measured: Sensitivities) -> Tensor:
    """Speedy-Splat's pruning score, as its rasterizer accumulates it."""
    return measured.speedy


__all__ = [
    "Derivatives",
    "Sensitivities",
    "derivatives",
    "projection_jacobian",
    "pup_score",
    "sensitivity_scores",
    "speedy_splat_score",
]
