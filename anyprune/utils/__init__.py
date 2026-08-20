"""
Shareds utility functions useful for more than one wrapper.
"""
import contextlib
import io
import logging


@contextlib.contextmanager
def _muted(enabled: bool):
    """
    Allows for muting verbose outputs of external submodules without
    changing their upstream source code.
    """
    if not enabled: yield; return
    previous_level = logging.root.manager.disable
    try: 
        logging.disable(logging.WARNING)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally: 
        logging.disable(previous_level)
        

__all__ = [
    "_muted",
]