"""
Utility code to run a feedforward 3DGS method.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Generator, Tensor

from ..gaussians import Gaussians
from ..utils import to_reconstruction_frame


@dataclass
class ViewSet:
    """
    A set of views of a scene: the images themselves and the cameras
    they were taken with, written in the frame of the Gaussians they go
    with.
    """
    images: Tensor     # (V, 3, H, W), in [0, 1]
    poses: Tensor      # (V, 4, 4), camera-to-world, OpenGL axes
    intrinsics: Tensor # (V, 3, 3), in pixels of 'images'

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def image_shape(self) -> Tuple[int, int]:
        return self.images.shape[-2:]

    def __add__(self, other: "ViewSet") -> "ViewSet":
        return ViewSet(
            images=torch.cat([self.images, other.images], dim=0),
            poses=torch.cat([self.poses, other.poses], dim=0),
            intrinsics=torch.cat([self.intrinsics, other.intrinsics], dim=0),
        )

    def __getitem__(self, index) -> "ViewSet":
        index = torch.as_tensor(index, device=self.images.device)
        return ViewSet(
            images=self.images[index],
            poses=self.poses[index],
            intrinsics=self.intrinsics[index],
        )

    def thin(
        self, num_views: int, generator: Optional[Generator] = None
    ) -> "ViewSet":
        """
        Randomly select 'num_views' of these views, one from each of 
        'num_views' equal stretches of the set, returning it unchanged 
        if it is already that small.
        """
        if len(self) <= num_views:
            return self
        edges = torch.linspace(0, len(self), num_views + 1).round().long()
        offsets = torch.rand(num_views, generator=generator)
        widths = (edges[1:] - edges[:-1]).clamp(min=1)
        picked = edges[:-1] + (offsets * widths).long().clamp(max=widths - 1)
        return self[picked]


@dataclass
class Reconstruction:
    """
    What a frozen reconstructor makes of a scene: one set of Gaussians
    and the two sets of views, the ones it saw and the ones held out.
    """
    gaussians: Gaussians
    context: ViewSet
    test: ViewSet

    def views(self, which: str) -> ViewSet:
        """
        The views to supervise or score on: the context views, the
        held-out ones, or both together.
        """
        if which == "context": return self.context
        if which == "test": return self.test
        if which == "all": return self.context + self.test
        raise ValueError(f"Unknown view set {which!r}, expected context, test or all")


def _downscale(images: Tensor, factor: int) -> Tensor:
    """
    Shrink a (V, 3, H, W) batch of images by a whole factor.
    """
    height, width = images.shape[-2:]
    assert height % factor == 0 and width % factor == 0, (
        f"Cannot downscale {height}x{width} images by {factor}"
    )
    return F.interpolate(
        images, size=(height // factor, width // factor),
        mode="bilinear", align_corners=False, antialias=True,
    )


def _rescale_intrinsics(intrinsics: Tensor, factor: int) -> Tensor:
    """
    Rewrite (V, 3, 3) pinhole matrices given in pixels of an image in
    pixels of the same image enlarged by a whole factor.
    """
    rescaled = intrinsics.clone()
    # Pixel centers sit at integer coordinates plus a half
    rescaled[:, :2, :2] *= factor
    rescaled[:, :2, 2] = (rescaled[:, :2, 2] + 0.5) * factor - 0.5
    return rescaled


@torch.no_grad()
def reconstruct(
    reconstructor: nn.Module,
    scene: Dict[str, Tensor],
    context_idx: Tensor,
    test_idx: Tensor,
    max_gaussians: Optional[int] = None,
    generator: Optional[Generator] = None,
    context_downscale: int = 1,
) -> Reconstruction:
    """
    Run a frozen feed-forward reconstructor on the context views of a
    scene and gather everything needed to render it.

    A reconstructor works in a canonical frame of its own, so the
    context cameras it predicts are used as they come while the held-out
    ones, which the dataset gives in its world frame, are converted to 
    that frame first.
    
    The predicted intrinsics are used for both,
    even though the dataset's are the more accurate ones, because those
    do not agree with the recovered frame and cost a lot of PSNR.
    """
    images = scene["images"]
    poses = scene["poses"].reshape(-1, 4, 4)

    context_images = images[context_idx]
    if context_downscale > 1:
        context_images = _downscale(context_images, context_downscale)
    pred_poses, pred_intrinsics, gaussians = reconstructor(context_images)
    if context_downscale > 1:
        # Predicted in pixels of what the model was shown, needed in
        # pixels of the images everything is scored against
        pred_intrinsics = _rescale_intrinsics(pred_intrinsics, context_downscale)
    if max_gaussians is not None:
        gaussians = gaussians.subsample(max_gaussians, generator=generator)

    test_poses = to_reconstruction_frame(
        poses[context_idx], pred_poses, poses[test_idx]
    )
    test_intrinsics = pred_intrinsics.mean(dim=0, keepdim=True)
    test_intrinsics = test_intrinsics.expand(len(test_idx), 3, 3)

    return Reconstruction(
        gaussians=gaussians,
        context=ViewSet(
            images=images[context_idx],
            poses=pred_poses,
            intrinsics=pred_intrinsics,
        ),
        test=ViewSet(
            images=images[test_idx],
            poses=test_poses,
            intrinsics=test_intrinsics,
        ),
    )


__all__ = [
    "Reconstruction",
    "ViewSet",
    "reconstruct",
]
