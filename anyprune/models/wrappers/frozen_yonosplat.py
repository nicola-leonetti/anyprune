"""
A PyTorch module to run YoNoSplat to produce Gaussians from a set of
views, all with a simplified interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dacite import from_dict
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import Tensor

from anyprune.utils import _muted
with _muted(True): from ..utils import (
    EncoderYoNoSplatCfg, YONOSPLAT_ROOT, build_yonosplat_encoder,
)
from ...gaussians import Gaussians


# The released checkpoints are all trained at 224x224 and only hold up
# at that resolution: on 448x448 the model does not work. 
NATIVE_IMAGE_SIZE = 224
# The side, in pixels, of the patches the backbone cuts images into
PATCH_SIZE = 14

# YoNoSplat is configured through Hydra, so rather than repeat its
# settings we read the ones it ships and apply the overrides its
# EVALUATION.md passes for the uncalibrated, unposed case.
_ENCODER_CONFIG = YONOSPLAT_ROOT / "config" / "model" / "encoder" / "yonosplat.yaml"
_BACKBONE_CONFIG = _ENCODER_CONFIG.parent / "backbone" / "local_global.yaml"
_UNPOSED_OVERRIDES = {
    # Predict the cameras rather than being handed the ground truth
    "pose_free": True,
    # Gradient checkpointing only pays off while training
    "use_checkpoint": False,
    # The whole encoder comes from the checkpoint we load ourselves
    "pretrained_weights": "",
    "backbone": {
        "predict_intrinsics": True,
        # Condition on the predicted intrinsics, as there are no
        # ground-truth ones to condition on
        "use_pred_intrinsics_for_embed": True,
    },
}


def _encoder_config() -> EncoderYoNoSplatCfg:
    """
    The configuration of YoNoSplat's encoder, typed the way YoNoSplat's
    own entry point types it.
    """
    cfg = OmegaConf.load(_ENCODER_CONFIG)
    # Hydra would resolve the 'defaults' list, we only need the backbone
    del cfg["defaults"]
    cfg.backbone = OmegaConf.load(_BACKBONE_CONFIG)
    cfg = OmegaConf.merge(cfg, _UNPOSED_OVERRIDES)
    return from_dict(EncoderYoNoSplatCfg, OmegaConf.to_container(cfg))


class FrozenYoNoSplat(nn.Module):
    def __init__(self, pretrained_ckpt: str, quiet: bool):
        super().__init__()
        self.quiet = quiet
        with _muted(quiet):
            self.model = build_yonosplat_encoder(_encoder_config())
            self._load_weights(pretrained_ckpt)
        self.model.eval()
        self.model.requires_grad_(False)

    def _load_weights(self, pretrained_ckpt: str):
        """
        Fill the encoder with the weights of a checkpoint, named as the
        file of a Hugging Face repository it lives in, such as
        'botaoye/YoNoSplat/dl3dv_224x224_ctx2to32.ckpt'. It is
        downloaded on first use and read from the cache afterwards.

        YoNoSplat has no from_pretrained() of its own to do this: its
        checkpoints are the ones PyTorch Lightning wrote while training,
        carrying the decoder and the losses as well, so the encoder's
        share of the weights has to be picked out of them.
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
        missing, _ = self.model.load_state_dict(state_dict, strict=False)
        assert not missing, (
            f"{len(missing)} weights missing from {pretrained_ckpt}, e.g. {missing[:3]}"
        )

    @staticmethod
    def _to_native_resolution(context_images: Tensor) -> Tensor:
        """
        Resize a (V, 3, H, W) batch of images to the resolution the
        encoder was trained on, keeping their aspect ratio and both
        sides a whole number of patches.

        What comes out of the encoder does not have to stay at this
        resolution: the Gaussians are in world space and the cameras are
        predicted in normalized coordinates, so both can be rendered at
        whatever size the caller's images are.
        """
        height, width = context_images.shape[-2:]
        scale = NATIVE_IMAGE_SIZE / max(height, width)
        native_shape = tuple(
            max(round(side * scale / PATCH_SIZE), 1) * PATCH_SIZE
            for side in (height, width)
        )
        if native_shape == (height, width):
            return context_images
        return F.interpolate(
            context_images, size=native_shape,
            mode="bilinear", align_corners=False, antialias=True,
        )

    @staticmethod
    def _placeholder_intrinsics(context_images: Tensor) -> Tensor:
        """
        The intrinsics YoNoSplat's backbone conditions on, of shape
        (1, V, 3, 3).

        The backbone always takes an intrinsic matrix, even when it is
        the one predicting it: it overwrites the focal lengths of the
        matrix it is handed with its own prediction and keeps the rest.
        All it ends up using of what we pass is therefore the principal
        point, which we put at the center of the image, and the
        normalized convention it is written in, first row over the image
        width and second over its height.
        """
        num_views = context_images.shape[0]
        intrinsics = torch.eye(3).to(context_images)
        intrinsics[0, 2] = intrinsics[1, 2] = 0.5
        return intrinsics.expand(1, num_views, 3, 3).contiguous()

    @staticmethod
    def _to_dl3dv_convention(
        visualization_dump: dict, image_shape: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        """
        Convert YoNoSplat's predicted cameras to the DL3DV convention
        the rest of the project uses, so that they can go straight into
        Gaussians.rasterize().

        YoNoSplat follows pixelSplat's convention, the same one AnySplat
        uses: 'pred_camera_poses' (B, V, 4, 4) are camera-to-world
        matrices with OpenCV axes (+Y down, +Z forwards), written in the
        frame of the first context view. DL3DV instead uses OpenGL
        camera axes (+Y up, +Z backwards), so we flip the Y and Z axes.

        Of the intrinsics only the focal lengths are predicted,
        'intrinsic_pred' (B * V, 2), divided by the image width and
        height. We scale them back up into pinhole matrices centered on
        the image, which is where the encoder was told the principal
        point is.
        """
        height, width = image_shape

        poses = visualization_dump["pred_camera_poses"][0]
        opencv_to_opengl = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]).to(poses))
        poses = poses @ opencv_to_opengl

        # The focal lengths come out of an autocast region, in half
        # precision, unlike everything else the encoder hands back
        focals = visualization_dump["intrinsic_pred"].float()
        intrinsics = torch.eye(3).to(poses).repeat(focals.shape[0], 1, 1)
        intrinsics[:, 0, 0] = focals[:, 0] * width
        intrinsics[:, 1, 1] = focals[:, 1] * height
        intrinsics[:, 0, 2] = 0.5 * width
        intrinsics[:, 1, 2] = 0.5 * height

        return poses, intrinsics

    def forward(self, context_images: Tensor):
        """
        Takes a (V, 3, H, W) tensor of images in [0, 1] and returns a
        tuple (poses, intrinsics, gaussians).

        Poses is a (V, 4, 4) tensor of camera-to-world matrices and
        intrinsics a (V, 3, 3) tensor of pinhole matrices, both already
        in the convention that Gaussians.rasterize() expects, and in
        pixels of the images passed in even though the reconstruction
        happens at the encoder's own resolution.
        """
        assert context_images.shape[1] == 3, f"Expected (V, 3, H, W) shape for context images, got {context_images.shape}"
        assert context_images.dim() == 4, f"context_images should have dim 4, got {context_images.dim()}"
        # The encoder works on batches of scenes, we do one at a time,
        # at the resolution its weights were trained for
        views = self._to_native_resolution(context_images)
        context = {
            "image": views.unsqueeze(0),
            "intrinsics": self._placeholder_intrinsics(views),
        }
        # The cameras only come out of the encoder through this dump,
        # what it returns are the Gaussians alone
        visualization_dump = {}
        with _muted(self.quiet):
            gaussians = self.model(
                context, global_step=0, visualization_dump=visualization_dump
            )
        poses, intrinsics = self._to_dl3dv_convention(
            visualization_dump, context_images.shape[-2:]
        )
        return poses, intrinsics, Gaussians.from_yonosplat(gaussians)