"""
Context manager used to import from more than one external model 
codebase.

More than one model (e.g. AnySplat and YoNoSplat) are both laid out as 
a top-level 'src' package and both import their own files by that name, 
so only one of them can answer to 'src' at a time. 
To solve that, owns_src() lends the name 'src' to one repository for as 
long as it takes to import from it, and parks the modules of the other 
one meanwhile.
"""
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator, Optional


EXTERNAL_ROOT = Path(__file__).resolve().parent / "external"

# The 'src' modules of every repository imported so far, keyed by the
# name of its directory under external/ and parked here for as long as
# another repository holds the name.
_parked: dict[str, dict[str, ModuleType]] = {}
_owner: Optional[str] = None


def _take_src_modules() -> dict[str, ModuleType]:
    """
    Remove the top-level 'src' package and everything under it from
    sys.modules, then returns all the taken modules so they can be
    restored later.
    """
    taken = {
        name: module for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    for name in taken:
        del sys.modules[name]
    return taken


@contextmanager
def owns_src(repository: str) -> Iterator[None]:
    """
    Import from `repository`, one of the codebases under external/, 
    linking the repo's files to the name 'src'.

    Everything that a repo needs has to be imported inside an owns_dir
    block.
    """
    global _owner
    previous_owner = _owner
    if previous_owner is not None:
        _parked[previous_owner] = _take_src_modules()
    sys.modules.update(_parked.pop(repository, {}))

    root = str(EXTERNAL_ROOT / repository)
    sys.path.insert(0, root)
    _owner = repository
    try:
        yield
    finally:
        sys.path.remove(root)
        _owner = previous_owner
        _parked[repository] = _take_src_modules()
        if previous_owner is not None:
            sys.modules.update(_parked.pop(previous_owner, {}))


__all__ = [
    "owns_src",
]