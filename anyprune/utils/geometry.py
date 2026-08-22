"""
Geometry helpers shared across the project: camera conventions, the
frames the different models reconstruct in, and the transforms between
them.
"""
import torch
from torch import Tensor


def to_reconstruction_frame(
    context_poses: Tensor, pred_context_poses: Tensor, test_poses: Tensor
) -> Tensor:
    """
    Express the (V, 4, 4) camera-to-world `test_poses` in the frame a
    model reconstructed in, so that they can be rasterized against the
    Gaussians it produced.

    A feed-forward model reconstructs in a canonical frame of its own,
    related to the dataset's world frame by an unknown similarity
    transform. The context views give us the correspondence to recover
    it, since `context_poses` and `pred_context_poses` are the same
    cameras written in the two frames.

    The rotation comes from the camera orientations rather than from
    their centers: a dense run of frames is close to a straight dolly,
    and fitting all seven degrees of freedom to centers that nearly lie
    on a line leaves the rotation about that line unconstrained. Each
    orientation instead pins down all three angles on its own. The
    centers are still what the scale and translation are read off, which
    they determine perfectly well.
    """
    source = context_poses[:, :3, 3]
    target = pred_context_poses[:, :3, 3]
    source_mean, target_mean = source.mean(dim=0), target.mean(dim=0)
    source_centered, target_centered = source - source_mean, target - target_mean

    # Orthogonal Procrustes on the orientations
    covariance = torch.einsum(
        "vij,vkj->ik", pred_context_poses[:, :3, :3], context_poses[:, :3, :3]
    )
    u, _, vt = torch.linalg.svd(covariance)
    # Keep the SVD from handing us a reflection instead of a rotation
    correction = torch.ones(3).to(source)
    if torch.det(u) * torch.det(vt) < 0:
        correction[-1] = -1.0
    rotation = u @ torch.diag(correction) @ vt

    scale = target_centered.norm() / source_centered.norm()
    translation = target_mean - scale * rotation @ source_mean

    # Only the camera centers pick up the scale, a rotation stays one
    transformed = test_poses.clone()
    transformed[:, :3, :3] = rotation @ test_poses[:, :3, :3]
    transformed[:, :3, 3] = test_poses[:, :3, 3] @ (scale * rotation).T + translation
    return transformed


__all__ = [
    "to_reconstruction_frame",
]
