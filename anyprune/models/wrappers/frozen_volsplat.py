"""
A PyTorch module to run VolSplat to produce Gaussians from a set of
posed views, all with a simplified interface.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import Tensor

from anyprune.utils import _muted
from ..utils import VOLSPLAT_ROOT, build_volsplat_encoder
from ...gaussians import Gaussians


# The side, in pixels, of the square frames the only released checkpoint
# (RealEstate10K) is trained and evaluated on
TRAINED_IMAGE_SIZE = 256
# Both sides have to be a whole number of the encoder's patches, which
# its data shim would otherwise crop them to (shim_patch_size *
# downscale_factor)
PATCH_SIZE = 16
# The depth range VolSplat's RealEstate10K configuration hands the
# encoder: it sweeps its depth candidates in it
NEAR, FAR = 0.5, 100.0
# The median depth, in the units of its poses, of what VolSplat predicts
# on the RealEstate10K test scenes it is evaluated on (6 views of its
# evaluation index), as the median over scenes: 3.02 on the 28 scenes of
# test chunks 0-1, 3.16 on the 69 of chunks 10-50. Its scales and its voxel grid are absolute lengths,
# so a scene can be brought to this median depth before it is
# reconstructed (FrozenVolSplat's 'target_median_depth'). On DL3DV it is
# not: the scenes at the scale of their poses (median depths of about
# 7-30) do 2-3 dB better than at 3 and the same as at 6-12, and above
# 12 the depths pile up against FAR and the search diverges (6 val
# scenes, 2/4/8 views).
TARGET_MEDIAN_DEPTH = 3.0
# How close the median depth has to land to the target, as a ratio,
# before the scene's scale is taken, and how many passes it gets there in
MEDIAN_DEPTH_TOLERANCE = 1.1
MAX_SCALE_PASSES = 4

# VolSplat is configured through Hydra, so rather than repeat its
# settings we read the encoder file it ships and apply what its
# RealEstate10K experiment and its README's evaluation command override
# on top of it.
_ENCODER_CONFIG = VOLSPLAT_ROOT / "config" / "model" / "encoder" / "volsplat.yaml"
_EXPERIMENT_CONFIG = VOLSPLAT_ROOT / "config" / "experiment" / "re10k.yaml"
_README_OVERRIDES = {
    "num_scales": 2,
    "upsample_factor": 2,
    "lowest_feature_resolution": 4,
    "monodepth_vit_type": "vitb",
    # The intermediate depth only feeds a training loss
    "supervise_intermediate_depth": False,
    # Only loaded at initialization, and the checkpoint holds them
    "unimatch_weights_path": None,
}

# OpenCV <-> OpenGL camera axes, the same matrix both ways
_FLIP_YZ = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def _encoder_config() -> dict:
    """
    The configuration of VolSplat's encoder for its RealEstate10K
    checkpoint.
    """
    cfg = OmegaConf.load(_ENCODER_CONFIG)
    experiment = OmegaConf.load(_EXPERIMENT_CONFIG).model.encoder
    return OmegaConf.to_container(OmegaConf.merge(cfg, experiment, _README_OVERRIDES))


class FrozenVolSplat(nn.Module):
    # Like ReSplat, VolSplat is handed the cameras of its context views
    # rather than predicting them
    needs_cameras = True

    def __init__(
        self,
        pretrained_ckpt: str,
        quiet: bool,
        image_size: Optional[int] = None,
        target_median_depth: Optional[float] = None,
    ):
        """
        'image_size' is the short side, in pixels, the frames are resized
        to before the encoder sees them, or None to show it the frames
        at their own resolution.

        The checkpoint is trained on frames of TRAINED_IMAGE_SIZE, but
        its Gaussians are about a pixel of what it was shown across, and
        the rasterizer's 0.3-pixel dilation closes the gaps between
        them: shown 256x256 frames and rendered at the 448x448 of this
        project, the field is full of holes (16.9 / 19.9 dB of PSNR on
        held-out views at 2 / 4 views, 6 val scenes; 23.4 / 28.6 dB
        rendered at 256). Shown the frames at their own 448x448 it gets
        26.9 / 29.6 dB, with 200k Gaussians per view.

        'target_median_depth' brings each scene to that median predicted
        depth before it is reconstructed (see TARGET_MEDIAN_DEPTH), and
        None takes it at the scale of its poses, in a single pass.
        """
        super().__init__()
        self.quiet = quiet
        self.image_size = image_size
        self.target_median_depth = target_median_depth
        with _muted(quiet):
            self.model = build_volsplat_encoder(_encoder_config())
            self._load_weights(pretrained_ckpt)
        self.model.eval()
        self.model.requires_grad_(False)
        # The Gaussians VolSplat's encoder hands back keep only their
        # covariances: the scales and rotations they were built from are
        # read off its adapter on the way out (which it calls through
        # .forward(), past any hook)
        self._adapted = None
        adapt = self.model.gaussian_adapter.forward

        def kept(*args, **kwargs):
            self._adapted = adapt(*args, **kwargs)
            return self._adapted

        self.model.gaussian_adapter.forward = kept

    def _load_weights(self, pretrained_ckpt: str):
        """
        Fill the encoder with the weights of a checkpoint, named as the
        file of a Hugging Face repository it lives in, such as
        'lhmd/VolSplat/volsplat-re10k-256x256.ckpt'. It is downloaded
        on first use and read from the cache afterwards.

        The checkpoints hold the state of VolSplat's Lightning wrapper,
        whose only weights are the encoder's.
        """
        repo_id, filename = pretrained_ckpt.rsplit("/", 1)
        checkpoint = torch.load(
            hf_hub_download(repo_id=repo_id, filename=filename), map_location="cpu"
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        prefix = "encoder."
        state_dict = {
            key[len(prefix):]: weights for key, weights in state_dict.items()
            if key.startswith(prefix)
        }
        assert state_dict, f"No encoder weights in {pretrained_ckpt}"
        self.model.load_state_dict(state_dict)

    def _native_shape(self, image_shape: tuple[int, int]) -> tuple[int, int]:
        """
        The shape the encoder is shown frames of 'image_shape' at: the
        short side brought to 'image_size' if there is one, keeping the
        aspect ratio, and both sides a whole number of patches.
        """
        scale = 1.0 if self.image_size is None else self.image_size / min(image_shape)
        return tuple(
            max(round(side * scale / PATCH_SIZE), 1) * PATCH_SIZE
            for side in image_shape
        )

    @staticmethod
    def _normalized_intrinsics(
        intrinsics: Tensor, image_shape: tuple[int, int], native_shape: tuple[int, int]
    ) -> Tensor:
        """
        Pinhole matrices in pixels of frames of 'image_shape' rewritten
        the way VolSplat takes them: for those frames resized to
        'native_shape', with the first row divided by the width and the
        second by the height.
        """
        normalized = intrinsics.clone()
        for row, (side, native_side) in enumerate(zip(image_shape, native_shape)):
            # Rows are x then y, sides height then width
            row = 1 - row
            scale = native_side / side
            # Pixel centers sit at integer coordinates plus a half
            normalized[:, row, row] *= scale
            normalized[:, row, 2] = (normalized[:, row, 2] + 0.5) * scale - 0.5
            normalized[:, row] /= native_side
        return normalized

    def _encode(self, views: Tensor, extrinsics: Tensor, intrinsics: Tensor):
        """
        One pass of the encoder over (V, 3, h, w) views with their
        (V, 4, 4) OpenCV camera-to-world extrinsics and normalized
        intrinsics: its Gaussians, the adapter's output they came from,
        and its (V, h, w) depths.
        """
        num_views = views.shape[0]
        context = {
            "image": views.unsqueeze(0),
            "extrinsics": extrinsics.unsqueeze(0),
            "intrinsics": intrinsics.unsqueeze(0),
            "near": torch.full((1, num_views), NEAR).to(views),
            "far": torch.full((1, num_views), FAR).to(views),
        }
        # MinkowskiEngine's hashing and sorting (thrust) cudaMalloc their
        # buffers past torch's caching allocator, so they only get what
        # torch is not holding cached: it is handed back first, and once
        # more before the one retry, and running out anyway is raised as
        # torch's own error, which callers know to catch
        for attempt in range(2):
            torch.cuda.empty_cache()
            try:
                with _muted(self.quiet):
                    output = self.model(context, global_step=0, deterministic=False)
                break
            except MemoryError as error:
                if "out of memory" not in str(error) or attempt == 1:
                    raise torch.cuda.OutOfMemoryError(f"VolSplat: {error}") from error
        adapted, self._adapted = self._adapted, None
        return output["gaussians"], adapted, output["depths"][0]

    def forward(self, context_images: Tensor, poses: Tensor, intrinsics: Tensor):
        """
        Takes a (V, 3, H, W) tensor of images in [0, 1], with their
        (V, 4, 4) camera-to-world poses and their (V, 3, 3) pinhole
        intrinsics in pixels of those images, both in the convention of
        Gaussians.rasterize(), and returns a tuple
        (poses, intrinsics, gaussians).

        The scene is put in the frame of the first context view and,
        with a 'target_median_depth', scaled until the median depth
        VolSplat predicts in it is that target: a pass at the scene's
        own scale, then one at the scale that would have hit the target,
        and so on until it lands within MEDIAN_DEPTH_TOLERANCE. The
        poses that come back are the ones passed in written in that
        frame and at that scale, and the intrinsics are the ones passed
        in.
        """
        assert context_images.shape[1] == 3, f"Expected (V, 3, H, W) shape for context images, got {context_images.shape}"
        assert context_images.dim() == 4, f"context_images should have dim 4, got {context_images.dim()}"
        num_views = context_images.shape[0]
        poses = poses.reshape(num_views, 4, 4).to(context_images)
        intrinsics = intrinsics.reshape(num_views, 3, 3).to(context_images)

        image_shape = tuple(context_images.shape[-2:])
        native_shape = self._native_shape(image_shape)
        views = context_images
        if native_shape != image_shape:
            views = F.interpolate(
                views, size=native_shape, mode="bilinear", align_corners=False, antialias=True,
            )
        normalized = self._normalized_intrinsics(intrinsics, image_shape, native_shape)

        # OpenGL -> OpenCV camera axes, then into the frame of the first
        # context view (in double precision: the context views of a
        # dense capture are centimeters apart, a long way from the origin)
        flip = _FLIP_YZ.to(poses.device, torch.float64)
        extrinsics = poses.double() @ flip
        extrinsics = torch.linalg.inv(extrinsics[0]) @ extrinsics

        scale = 1.0
        for _ in range(MAX_SCALE_PASSES):
            scaled = extrinsics.clone()
            scaled[:, :3, 3] *= scale
            scaled = scaled.to(poses.dtype)
            gaussians, adapted, depths = self._encode(views, scaled, normalized)
            median_depth = depths.median().item()
            if self.target_median_depth is None:
                break
            ratio = self.target_median_depth / median_depth
            if max(ratio, 1.0 / ratio) <= MEDIAN_DEPTH_TOLERANCE:
                break
            scale *= ratio
        self.last_scale, self.last_median_depth = scale, median_depth

        gaussians = Gaussians(
            means=gaussians.means[0],
            covariances=gaussians.covariances[0],
            harmonics=gaussians.harmonics[0],
            opacities=gaussians.opacities[0],
            # Unlike YoNoSplat's and ReSplat's, its rotations are written
            # in the world frame, as its covariances are
            scales=adapted.scales.reshape(-1, 3),
            rotations=adapted.rotations.reshape(-1, 4),
        )
        return scaled @ flip.to(scaled), intrinsics, gaussians
