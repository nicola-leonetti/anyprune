"""
Definitions borrowed from external model codebases (e.g. AnySplat), to
be used elsewhere in the project.

The codebases are imported here side-by-side using the tricks defined in 
_external.py and their methods are exposed for the rest of the project
to use.
"""
from pathlib import Path

import gin

from ._external import (
    EXTERNAL_ROOT, owns, shim_torch_attention, stub_unbuilt_extensions,
)

with owns("AnySplat"):
    from src.model.model import AnySplat
    from src.model.types import Gaussians as AnySplatGaussians
    from src.model.encoder.common.gaussians import build_covariance as build_anysplat_covariance
    from src.utils.image import process_image

# YoNoSplat is written against a newer PyTorch than this project pins,
# and reaches its decoder's rasterizer on the way to its encoder even
# though the encoder is all we ever run.
shim_torch_attention()
stub_unbuilt_extensions("diff_gaussian_rasterization")
with owns("YoNoSplat"):
    from src.model.encoder import EncoderYoNoSplat, EncoderYoNoSplatCfg
    from src.model.encoder import get_encoder as _get_yonosplat_encoder
    from src.model.types import Gaussians as YoNoSplatGaussians

# SplatFormer's backbone lives in Pointcept, which imports every model
# it ships as soon as it is touched, several of them built on CUDA
# extensions PointTransformerV3 never calls.
stub_unbuilt_extensions("pointops", "pointops2", "pointgroup_ops")
with owns("SplatFormer"):
    # Imported together so that they agree on one FeaturePredictor
    # class: build_optimizer branches on its exact identity
    from models.feature_predictor import FeaturePredictor as SplatFormerFeaturePredictor
    from utils.loss_utils import lpips_loss_fn
    from utils.optimizers import build_optimizer as _build_splatformer_optimizer
    from utils.optimizers import build_scheduler as build_splatformer_scheduler
    from utils.transform_utils import MinMaxScaler


# Where YoNoSplat keeps the configuration files its encoder is built
# from, as it has no pre-configured entry point of its own.
YONOSPLAT_ROOT = EXTERNAL_ROOT / "YoNoSplat"

# The gin file describing the SplatFormer architecture the released
# checkpoints were trained with, PointTransformerV3 at SH degree 1.
SPLATFORMER_ROOT = EXTERNAL_ROOT / "SplatFormer"
SPLATFORMER_MODEL_CONFIG = SPLATFORMER_ROOT / "configs" / "model" / "ptv3.gin"


def build_anysplat(pretrained_ckpt: str) -> AnySplat:
    """
    Build AnySplat from a checkpoint, given as the id of a Hugging Face
    repository such as 'lhjiang/anysplat'.
    """
    with owns("AnySplat"):
        return AnySplat.from_pretrained(pretrained_ckpt)


def build_yonosplat_encoder(cfg: EncoderYoNoSplatCfg) -> EncoderYoNoSplat:
    """
    Build YoNoSplat's encoder, the half of it that turns images into
    Gaussians and cameras, with its weights left at initialization.
    """
    with owns("YoNoSplat"):
        encoder, _ = _get_yonosplat_encoder(cfg)
    return encoder


def build_splatformer(
    config_file: Path = SPLATFORMER_MODEL_CONFIG,
) -> SplatFormerFeaturePredictor:
    """
    Build SplatFormer's feature predictor with its weights left at
    initialization, configured by one of its own gin files.

    SplatFormer is configured through gin rather than through arguments,
    so instead of repeating the architecture we read the file its
    released checkpoints were trained with. The one setting we do not
    take from it is the checkpoint to resume: the weights are loaded by
    the caller, which knows where they live on this machine.
    """
    with owns("SplatFormer"):
        gin.parse_config_file(str(config_file))
        return SplatFormerFeaturePredictor(resume_ckpt=None)


def build_splatformer_optimizer(
    model: SplatFormerFeaturePredictor,
    lr_dict: dict,
    optimizer_type: str = "adam",
    optimizer_params: dict = {"eps": 1e-15},
):
    """
    Build the optimizer SplatFormer trains with, which gives the
    backbone and each of the output heads a learning rate of its own.
    """
    with owns("SplatFormer"):
        return _build_splatformer_optimizer(
            model,
            lr_dict=lr_dict,
            optimizer_type=optimizer_type,
            optimizer_params=optimizer_params,
        )


__all__ = [
    "AnySplatGaussians",
    "EncoderYoNoSplatCfg",
    "MinMaxScaler",
    "SPLATFORMER_MODEL_CONFIG",
    "SPLATFORMER_ROOT",
    "SplatFormerFeaturePredictor",
    "YONOSPLAT_ROOT",
    "YoNoSplatGaussians",
    "build_anysplat",
    "build_anysplat_covariance",
    "build_splatformer",
    "build_splatformer_optimizer",
    "build_splatformer_scheduler",
    "build_yonosplat_encoder",
    "lpips_loss_fn",
    "process_image",
]
