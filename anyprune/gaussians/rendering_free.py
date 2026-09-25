"""
A pruning score that renders nothing: REFINE's (Chen et al., ECCV 2026,
arXiv:2606.09074, anyprune/gaussians/external/REFINE), which reads the
field and where the cameras stand, and no image, rasterizer or depth
order.

Its released code (REFINE_pruning.py) weighs every Gaussian by how
close it stands to the cameras,

    H_i = o_i / C * sum_c 1 / (||x_i - c||^2 + 0.5)

over the C camera centres, and multiplies that by three attributes,
each divided by its mean over the field: the largest area two of its
axes span, the square of its luma, and the square of its opacity,

    S_i = w_geo * H_i A_i / mean(H A) + w_color * H_i Y_i^2 / mean(H Y^2)
          + w_opa * H_i o_i^2 / mean(H o^2)

with the w a mix set per scene from the spread of those attributes,
against constants the authors fit on optimized 3DGS scenes. It then
prunes the lowest share, one shot. The paper writes the proximity as
o_i / ((z_i^v)^2 + 0.05) with z the depth in each view (its eq. 19);
the code, which is what is run here, takes the distance to the camera
centre and never asks which way a camera looks, so a Gaussian behind
every camera counts as much as one in front. The 0.5 is in the units
of the scene, and a predicted field is in the reconstructor's
normalized ones rather than a COLMAP scan's.

Both functions that compute the score are REFINE's own, imported from
the submodule; only the field is handed to them in the shape they read.
"""
import functools
import random
from types import SimpleNamespace

import torch
from torch import Tensor

from .gaussians import Gaussians

# How many cameras REFINE reads at most, the rest dropped at random
# (load_cameras' sample_limit, --camera_limit)
REFINE_CAMERA_LIMIT = 64


@functools.cache
def _refine():
    """REFINE_pruning out of the submodule."""
    from ..models._external import owns

    with owns("REFINE"):
        import REFINE_pruning
    return REFINE_pruning


@torch.no_grad()
def refine_score(gaussians: Gaussians, poses: Tensor) -> Tensor:
    """
    REFINE's importance of every Gaussian, larger where it matters
    more, from the (V, 4, 4) camera-to-world poses of the views the
    reconstructor was given: only their centres are read.
    """
    refine = _refine()
    centres = poses.reshape(-1, 4, 4)[:, :3, 3].to(gaussians.means).float()
    if centres.shape[0] > REFINE_CAMERA_LIMIT:
        # The draw load_cameras makes of a larger set
        kept = random.Random(42).sample(range(centres.shape[0]), REFINE_CAMERA_LIMIT)
        centres = centres[kept]

    # The attributes GaussianModel reads off a .ply, activated as its
    # __init__ activates them, colour included; the opacity as the
    # rasterizer reads it, which is also the sigmoid's range
    sh_dc = gaussians.harmonics[..., 0].float()
    field = SimpleNamespace(
        xyz=gaussians.means.float(),
        opacity=gaussians.opacities.float().clamp(0.0, 1.0),
        scales=gaussians.scales.float(),
        sh_dc=sh_dc,
        rgb=torch.clamp(sh_dc * 0.282 + 0.5, 0.0, 1.0),
    )
    weights = refine.GaussianModel.get_scene_adaptive_weights(field)
    scores = refine.compute_rahd_scores_fast(
        field, [{"pos": centre} for centre in centres], weights,
    )
    # Out of inference mode, so that the caller can do with it what it
    # does with any other score
    return scores.clone()


__all__ = [
    "REFINE_CAMERA_LIMIT",
    "refine_score",
]
