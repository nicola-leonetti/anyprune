"""
Per-scene optimization of a field of Gaussians against the views it was
predicted from: the "fine-tuning" step the pruning papers put after
their cut (PUP 3D-GS prunes an optimized scene and fine-tunes what is
left; Speedy-Splat and RadSplat prune inside the optimization and keep
optimizing), transplanted to a feed-forward field.

The recipe is 3DGS's own, with densification off, since a pruning is
being measured: Adam over the means, log scales, rotations, opacity
logits and harmonics, a photometric loss of L1 and D-SSIM at 0.8 / 0.2
on one view a step, the learning rates of the reference implementation
with the mean's scaled by the scene's extent and decayed exponentially
to a hundredth of itself over the schedule. The context views are all
the supervision a feed-forward field ever had, and they are what the
optimization sees; the held-out half stays held out.

The scales and rotations are taken from the covariances rather than
from the fields that carry them, so that a field whatever model it came
from (or through) is optimized as the same thing it is rasterized as.
"""
import math
from dataclasses import replace
from typing import Dict, Iterator, Optional, Sequence, Tuple

import gsplat
import torch
import torch.nn.functional as F
from torch import Generator, Tensor

from ..evaluation.metrics import ssim
from .gaussians import Gaussians, view_matrices

# 3DGS's learning rates, with the mean's per unit of scene extent
LR_MEANS_PER_EXTENT = 1.6e-4
LR_MEANS_FINAL_RATIO = 0.01
LR_SCALES = 5e-3
LR_ROTATIONS = 1e-3
LR_OPACITIES = 0.05
LR_HARMONICS_DC = 2.5e-3
LR_HARMONICS_REST = 2.5e-3 / 20
D_SSIM_WEIGHT = 0.2
# 3DGS's scene extent is the radius of the cameras' bounding sphere,
# padded; a feed-forward capture's cameras can sit within a step of
# each other, so the extent is at least the median distance of the
# field from them
EXTENT_PADDING = 1.1


def _factorize(covariances: Tensor) -> Tuple[Tensor, Tensor]:
    """
    (N, 3) scales and (N, 4) unit quaternions, wxyz as gsplat takes
    them, of the rotations that turn diag(scales^2) into the given (N,
    3, 3) covariances, by eigendecomposition.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances.double())
    scales = eigenvalues.clamp_min(1e-16).sqrt()
    # A proper rotation: an eigenbasis of negative determinant has one
    # axis flipped
    flip = torch.linalg.det(eigenvectors) < 0
    eigenvectors[flip, :, 2] *= -1
    rotations = _matrix_to_quaternion(eigenvectors)
    return scales.float(), rotations.float()


def _matrix_to_quaternion(matrices: Tensor) -> Tensor:
    """(N, 4) wxyz unit quaternions of (N, 3, 3) rotation matrices (Shepperd's method)."""
    m = matrices
    diagonal = torch.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], dim=-1)
    trace = diagonal.sum(dim=-1)
    # The four candidates, each safe where its pivot is the largest
    candidates = torch.stack([
        torch.stack([1 + trace, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], dim=-1),
        torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + 2 * m[:, 0, 0] - trace, m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], dim=-1),
        torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 + 2 * m[:, 1, 1] - trace, m[:, 1, 2] + m[:, 2, 1]], dim=-1),
        torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 + 2 * m[:, 2, 2] - trace], dim=-1),
    ], dim=1)                                                                # (N, 4, 4)
    pivots = torch.stack([trace, diagonal[:, 0], diagonal[:, 1], diagonal[:, 2]], dim=-1)
    chosen = candidates[torch.arange(m.shape[0], device=m.device), pivots.argmax(dim=-1)]
    return F.normalize(chosen, dim=-1)


def _quaternion_to_matrix(wxyz: Tensor) -> Tensor:
    w, x, y, z = F.normalize(wxyz, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def scene_extent(gaussians: Gaussians, poses: Tensor) -> float:
    """
    The radius 3DGS scales the mean's learning rate by: of the cameras'
    bounding sphere, padded, or the median distance of the field from
    the cameras' centre when they sit closer than that.
    """
    centres = poses.reshape(-1, 4, 4)[:, :3, 3].float()
    middle = centres.mean(dim=0)
    cameras = (centres - middle).norm(dim=-1).max().item() * EXTENT_PADDING
    field = (gaussians.means.float() - middle).norm(dim=-1).median().item()
    return max(cameras, field, 1e-3)


def fine_tune(
    gaussians: Gaussians,
    poses: Tensor,
    intrinsics: Tensor,
    images: Tensor,
    steps: int,
    generator: Optional[Generator] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    factorize: bool = True,
) -> Gaussians:
    """
    The field after 'steps' steps of 3DGS optimization against the V
    'images' (V, 3, H, W) taken from 'poses' and 'intrinsics', given the
    way Gaussians.rasterize() takes them, one view a step drawn with
    'generator' (a CPU one). Nothing is added or removed: the count and
    the order of the field are what they were.
    """
    assert steps >= 0, f"Cannot optimize for {steps} steps"
    if steps == 0:
        return gaussians
    for _, field in fine_tune_stages(
        gaussians, poses, intrinsics, images, [steps], generator, near_plane, far_plane, factorize,
    ):
        return field


def fine_tune_stages(
    gaussians: Gaussians,
    poses: Tensor,
    intrinsics: Tensor,
    images: Tensor,
    stages: Sequence[int],
    generator: Optional[Generator] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    factorize: bool = True,
) -> Iterator[Tuple[int, Gaussians]]:
    """
    fine_tune() handing the field out along the way: one (step, field)
    after each of the 'stages' (ascending step counts), the schedule
    being that of the last one, so that one optimization answers for
    every length up to it.

    With 'factorize' off the scales and rotations (xyzw) the field
    carries are optimized as they are, rather than the ones its
    covariances factorize into, so that what comes back is written in
    the same parameters as what went in: what a loss between the two
    needs.
    """
    stages = sorted(set(int(stage) for stage in stages))
    assert stages and stages[0] > 0, f"The stages have to be positive step counts, got {stages}"
    steps = stages[-1]
    device = gaussians.device
    poses = poses.reshape(-1, 4, 4).to(gaussians.means)
    intrinsics = intrinsics.reshape(-1, 3, 3).to(gaussians.means)
    viewmats = view_matrices(poses)
    num_views, height, width = images.shape[0], images.shape[-2], images.shape[-1]
    sh_degree = gaussians.sh_degree

    with torch.no_grad():
        if factorize:
            scales, rotations = _factorize(gaussians.covariances.float())
        else:
            scales = gaussians.scales.float().clamp_min(1e-8)
            rotations = F.normalize(gaussians.rotations.float(), dim=-1)[:, [3, 0, 1, 2]]
        opacities = gaussians.opacities.float().clamp(1e-4, 1 - 1e-4)
        parameters = {
            "means": gaussians.means.float().clone(),
            "log_scales": scales.log(),
            "rotations": rotations,
            "opacity_logits": torch.logit(opacities),
            "harmonics_dc": gaussians.harmonics.float()[:, :, :1].transpose(-2, -1).contiguous(),
            "harmonics_rest": gaussians.harmonics.float()[:, :, 1:].transpose(-2, -1).contiguous(),
        }
    for value in parameters.values():
        value.requires_grad_(True)
    extent = scene_extent(gaussians, poses)
    rates = {
        "means": LR_MEANS_PER_EXTENT * extent, "log_scales": LR_SCALES, "rotations": LR_ROTATIONS,
        "opacity_logits": LR_OPACITIES, "harmonics_dc": LR_HARMONICS_DC, "harmonics_rest": LR_HARMONICS_REST,
    }
    optimizer = torch.optim.Adam(
        [{"params": [value], "lr": rates[name], "name": name} for name, value in parameters.items()],
        eps=1e-15,
    )
    decay = LR_MEANS_FINAL_RATIO ** (1.0 / steps)

    for step in range(steps):
        view = int(torch.randint(num_views, (1,), generator=generator).item())
        colors = torch.cat([parameters["harmonics_dc"], parameters["harmonics_rest"]], dim=1)
        rendered, _, _ = gsplat.rasterization(
            means=parameters["means"],
            quats=parameters["rotations"],
            scales=parameters["log_scales"].exp(),
            opacities=torch.sigmoid(parameters["opacity_logits"]),
            colors=colors,
            viewmats=viewmats[view:view + 1], Ks=intrinsics[view:view + 1],
            width=width, height=height, near_plane=near_plane, far_plane=far_plane,
            sh_degree=sh_degree,
        )
        rendered = rendered.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        target = images[view:view + 1].float()
        loss = (1 - D_SSIM_WEIGHT) * (rendered - target).abs().mean() + D_SSIM_WEIGHT * (1 - ssim(rendered, target).mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for group in optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] *= decay
        if step + 1 in stages:
            yield step + 1, assemble(gaussians, parameters)


@torch.no_grad()
def assemble(gaussians: Gaussians, parameters: Dict[str, Tensor]) -> Gaussians:
    """The field the optimized parameters make, in the dtypes of the one they came from."""
    scales = parameters["log_scales"].exp()
    rotation = _quaternion_to_matrix(parameters["rotations"])
    scaled = rotation * scales.unsqueeze(-2)
    covariances = scaled @ scaled.transpose(-1, -2)
    harmonics = torch.cat([parameters["harmonics_dc"], parameters["harmonics_rest"]], dim=1).transpose(-2, -1)
    wxyz = F.normalize(parameters["rotations"], dim=-1)
    return replace(
        gaussians,
        means=parameters["means"].detach().to(gaussians.means.dtype),
        covariances=covariances.to(gaussians.covariances.dtype),
        harmonics=harmonics.contiguous().to(gaussians.harmonics.dtype),
        opacities=torch.sigmoid(parameters["opacity_logits"]).detach().to(gaussians.opacities.dtype),
        scales=scales.to(gaussians.scales.dtype),
        rotations=wxyz[:, [1, 2, 3, 0]].to(gaussians.rotations.dtype),
    )


__all__ = ["fine_tune", "fine_tune_stages", "scene_extent"]
