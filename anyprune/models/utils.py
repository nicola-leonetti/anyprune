"""
Definitions borrowed from external model codebases (e.g. AnySplat), to
be used elsewhere in the project.

This is the one place where those codebases are imported, since they
cannot be imported side by side without care: see owns_src() for the 
exact trick needed to achieve it.
"""
from ._external import EXTERNAL_ROOT, owns_src

with owns_src("AnySplat"):
    from src.model.model import AnySplat
    from src.model.types import Gaussians as AnySplatGaussians
    from src.model.encoder.common.gaussians import build_covariance as build_anysplat_covariance
    from src.utils.image import process_image

with owns_src("YoNoSplat"):
    from src.model.encoder import EncoderYoNoSplat, EncoderYoNoSplatCfg
    from src.model.encoder import get_encoder as _get_yonosplat_encoder
    from src.model.types import Gaussians as YoNoSplatGaussians


# Where YoNoSplat keeps the configuration files its encoder is built
# from, as it has no pre-configured entry point of its own.
YONOSPLAT_ROOT = EXTERNAL_ROOT / "YoNoSplat"


def build_anysplat(pretrained_ckpt: str) -> AnySplat:
    """
    Build AnySplat from a checkpoint, given as the id of a Hugging Face
    repository such as 'lhjiang/anysplat'.
    """
    with owns_src("AnySplat"):
        return AnySplat.from_pretrained(pretrained_ckpt)


def build_yonosplat_encoder(cfg: EncoderYoNoSplatCfg) -> EncoderYoNoSplat:
    """
    Build YoNoSplat's encoder, the half of it that turns images into
    Gaussians and cameras, with its weights left at initialization.
    """
    with owns_src("YoNoSplat"):
        encoder, _ = _get_yonosplat_encoder(cfg)
    return encoder


__all__ = [
    "AnySplatGaussians",
    "EncoderYoNoSplatCfg",
    "YONOSPLAT_ROOT",
    "YoNoSplatGaussians",
    "build_anysplat",
    "build_anysplat_covariance",
    "build_yonosplat_encoder",
    "process_image",
]