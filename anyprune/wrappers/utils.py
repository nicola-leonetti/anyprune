"""
Shareds utility functions useful for more than one wrapper.
"""
import contextlib
import io


@contextlib.contextmanager
def _muted(enabled: bool):
    """
    Allows for muting verbose outputs of external submodules without
    changing their upstream source code.
    """
    if not enabled: 
        yield; return
    with contextlib.redirect_stdout(io.StringIO()): 
        yield
        