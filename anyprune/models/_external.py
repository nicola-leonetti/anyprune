"""
Context manager used to import from more than one external model
codebase.

The codebases under external/ are not written to live side by side:
AnySplat and YoNoSplat are both laid out as a top-level 'src' package
and both import their own files by that name, while SplatFormer claims
the equally generic 'models', 'utils' and 'dataset'. Only one repository
can answer to a given name at a time.
To solve that, owns() lends its top-level names to one repository for as
long as it takes to import from it, and parks the modules of the others
meanwhile.
"""
import enum
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator, Optional, Tuple

import torch


EXTERNAL_ROOT = Path(__file__).resolve().parent / "external"

# The top-level packages every external codebase brings with it, and
# which owns() therefore has to hand over and take back.
_TOP_LEVEL_PACKAGES: dict[str, Tuple[str, ...]] = {
    "AnySplat": ("src",),
    "YoNoSplat": ("src",),
    "SplatFormer": ("models", "utils", "dataset"),
}

# The top-level modules of every repository imported so far, keyed by
# the name of its directory under external/ and parked here for as long
# as another repository holds those names.
_parked: dict[str, dict[str, ModuleType]] = {}
_owner: Optional[str] = None


def _take_modules(packages: Tuple[str, ...]) -> dict[str, ModuleType]:
    """
    Remove the given top-level packages and everything under them from
    sys.modules, then return all the taken modules so they can be
    restored later.
    """
    taken = {
        name: module for name, module in sys.modules.items()
        if any(name == package or name.startswith(f"{package}.") for package in packages)
    }
    for name in taken:
        del sys.modules[name]
    return taken


@contextmanager
def owns(repository: str) -> Iterator[None]:
    """
    Import from `repository`, one of the codebases under external/,
    linking the repo's files to the top-level names it expects.

    Everything that a repo needs has to be imported inside an owns()
    block, and so does anything that imports lazily at call time.
    """
    global _owner
    assert repository in _TOP_LEVEL_PACKAGES, (
        f"Unknown external repository {repository!r}, "
        f"expected one of {sorted(_TOP_LEVEL_PACKAGES)}"
    )
    previous_owner = _owner
    if previous_owner is not None:
        _parked[previous_owner] = _take_modules(_TOP_LEVEL_PACKAGES[previous_owner])
    sys.modules.update(_parked.pop(repository, {}))

    root = str(EXTERNAL_ROOT / repository)
    sys.path.insert(0, root)
    _owner = repository
    try:
        yield
    finally:
        sys.path.remove(root)
        _owner = previous_owner
        _parked[repository] = _take_modules(_TOP_LEVEL_PACKAGES[repository])
        if previous_owner is not None:
            sys.modules.update(_parked.pop(previous_owner, {}))


def stub_unbuilt_extensions(*names: str):
    """
    Stand in for compiled extensions that are not installed, so that a
    package which imports them eagerly can still be imported.
    """
    for name in names:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = _UnbuiltExtension(name)


def shim_torch_attention():
    """
    Provide torch.nn.attention on the versions of PyTorch that predate
    it, which is where it moved out of torch.backends.cuda.
    """
    if hasattr(torch.nn, "attention"):
        return

    class SDPBackend(enum.Enum):
        MATH = 0
        FLASH_ATTENTION = 1
        EFFICIENT_ATTENTION = 2

    @contextmanager
    def sdpa_kernel(backends) -> Iterator[None]:
        if not isinstance(backends, (list, tuple)):
            backends = [backends]
        with torch.backends.cuda.sdp_kernel(
            enable_flash=SDPBackend.FLASH_ATTENTION in backends,
            enable_math=SDPBackend.MATH in backends,
            enable_mem_efficient=SDPBackend.EFFICIENT_ATTENTION in backends,
        ):
            yield

    module = ModuleType("torch.nn.attention")
    module.SDPBackend = SDPBackend
    module.sdpa_kernel = sdpa_kernel
    sys.modules["torch.nn.attention"] = module
    # Bound on torch.nn as well, since the call sites reach it through
    # the package rather than through the name they imported
    torch.nn.attention = module


class _UnbuiltExtension(ModuleType):
    def __getattr__(self, attribute: str):
        # The module machinery makes hasattr() calls for dunders on the
        # way past, and those have to come back negative rather than
        # handing out a placeholder
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        # Anything else is handed a placeholder rather than refused, so
        # that a module which imports names off this one still imports.
        # It is calling one that fails, which nothing on the paths we
        # run ever does.
        return _unbuilt_symbol(f"{self.__name__}.{attribute}")


def _unbuilt_symbol(name: str):
    class Unbuilt:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"'{name}' was called but '{name.split('.')[0]}' is not "
                f"installed: build it if you need this code path"
            )
    Unbuilt.__name__ = name.rsplit(".", 1)[-1]
    return Unbuilt


__all__ = [
    "EXTERNAL_ROOT",
    "owns",
    "shim_torch_attention",
    "stub_unbuilt_extensions",
]
