"""
Context manager used to import from more than one external model
codebase.

The codebases under external/ are not written to live side by side:
AnySplat, YoNoSplat, ReSplat and VolSplat are all laid out as a top-level 'src' package
and all import their own files by that name, while SplatFormer claims
the equally generic 'models', 'utils' and 'dataset'. Only one repository
can answer to a given name at a time.
To solve that, owns() lends its top-level names to one repository for as
long as it takes to import from it, and parks the modules of the others
meanwhile. The pruning codebases under anyprune/gaussians/external/ that
this project imports from, LightGaussian and REFINE, are lent their
names the same way: LightGaussian also claims 'utils'.
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
    "ReSplat": ("src",),
    "VolSplat": ("src",),
    "SplatFormer": ("models", "utils", "dataset"),
    "LightGaussian": (
        "arguments", "gaussian_renderer", "lpipsPyTorch", "prune", "scene",
        "utils", "vectree",
    ),
    "REFINE": ("REFINE_pruning",),
}

# Where a repository lives when it is not under external/ here
_ROOTS: dict[str, Path] = {
    "LightGaussian": EXTERNAL_ROOT.parents[1] / "gaussians" / "external" / "LightGaussian",
    "REFINE": EXTERNAL_ROOT.parents[1] / "gaussians" / "external" / "REFINE",
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

    root = str(_ROOTS.get(repository, EXTERNAL_ROOT / repository))
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


@contextmanager
def standing_in(name: str, module: ModuleType) -> Iterator[None]:
    """
    Answer imports of the top-level module 'name' with 'module' for as
    long as the block runs, and put back whatever answered before.

    For a codebase that imports a compiled extension at module level
    and only needs a few of its functions, which 'module' provides: the
    codebase keeps the reference it bound while importing, and the
    others go on seeing what they saw.
    """
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous


def unshadow_minkowski_engine():
    """
    Import MinkowskiEngine, if it is installed, and take back off
    sys.path the directory it appends to it.

    It puts its own directory on the path to import its files by their
    bare names, and that directory holds a regular package called
    'utils', which then wins over LightGaussian's 'utils', a namespace
    package (no __init__.py), wherever the two sit on the path. Its files
    are all imported by then, so the names it looks up later are found
    in sys.modules. Pointcept imports it whenever it is installed, so
    this has to run before SplatFormer is imported.
    """
    try:
        import MinkowskiEngine
    except ImportError:
        return
    directory = str(Path(MinkowskiEngine.__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry).resolve()) != directory]


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
    "standing_in",
    "stub_unbuilt_extensions",
    "unshadow_minkowski_engine",
]
