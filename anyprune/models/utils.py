"""
Definitions borrowed from external model codebases (e.g. AnySplat), to
be used elsewhere in the project.

The codebases are imported here side-by-side using the tricks defined in 
_external.py and their methods are exposed for the rest of the project
to use.
"""
from pathlib import Path

import gin

from . import _pointops
from ._external import (
    EXTERNAL_ROOT, owns, shim_torch_attention, standing_in,
    stub_unbuilt_extensions, unshadow_minkowski_engine,
)

# Before anything imports MinkowskiEngine (Pointcept does, VolSplat
# needs it): see unshadow_minkowski_engine()
unshadow_minkowski_engine()

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

# ReSplat's point transformers import the pointops extension, which it
# ships as sources to compile, for a k-nearest-neighbour query alone:
# they are handed a PyTorch stand-in instead (_pointops.py).
with owns("ReSplat"), standing_in("pointops", _pointops.as_module()):
    from src.model.encoder import EncoderReSplat, EncoderReSplatCfg
    from src.model.encoder import get_encoder as _get_resplat_encoder
    from src.model.types import Gaussians as ReSplatGaussians

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

# Where ReSplat keeps the configuration files its encoder is built from.
RESPLAT_ROOT = EXTERNAL_ROOT / "ReSplat"

# Where VolSplat keeps the configuration files its encoder is built from.
VOLSPLAT_ROOT = EXTERNAL_ROOT / "VolSplat"

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


def build_resplat_encoder(cfg: EncoderReSplatCfg) -> EncoderReSplat:
    """
    Build ReSplat's encoder, which predicts the Gaussians and refines
    them, with its weights left at initialization.
    """
    with owns("ReSplat"):
        encoder, _ = _get_resplat_encoder(cfg)
    return encoder


def build_volsplat_encoder(cfg: dict):
    """
    Build VolSplat's encoder from the plain dictionary of its
    configuration, typed with its own configuration class, with its
    weights left at initialization.

    Unlike the other codebases it is only imported here, on first use:
    its sparse 3D decoder needs MinkowskiEngine, a compiled extension
    nothing else in the project does.
    """
    import MinkowskiEngine
    from dacite import from_dict

    # Its coordinate maps go through torch's caching allocator rather
    # than their own cudaMalloc (its thrust calls still do not: see
    # FrozenVolSplat._encode)
    MinkowskiEngine.set_gpu_allocator(MinkowskiEngine.GPUMemoryAllocatorType.PYTORCH)
    with owns("VolSplat"):
        from src.model.encoder import EncoderVolSplatCfg, get_encoder
        encoder, _ = get_encoder(from_dict(EncoderVolSplatCfg, cfg))
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
    "EncoderReSplatCfg",
    "MinMaxScaler",
    "RESPLAT_ROOT",
    "VOLSPLAT_ROOT",
    "ReSplatGaussians",
    "SPLATFORMER_MODEL_CONFIG",
    "SPLATFORMER_ROOT",
    "SplatFormerFeaturePredictor",
    "YONOSPLAT_ROOT",
    "YoNoSplatGaussians",
    "build_anysplat",
    "build_anysplat_covariance",
    "build_resplat_encoder",
    "build_volsplat_encoder",
    "build_splatformer",
    "build_splatformer_optimizer",
    "build_splatformer_scheduler",
    "build_yonosplat_encoder",
    "lpips_loss_fn",
    "process_image",
]
