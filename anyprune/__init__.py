from .utils import _muted
with _muted(True): from .models.external.AnySplat.src.utils.image import process_image


__all__ = [
    "process_image",
]
