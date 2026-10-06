"""
Telling a card that ran out of room apart from a run that went wrong.

Whatever is caught as an exhausted card is also said out loud, on the
console's error stream, where it was raised and what the card held,
however the caller goes on (a retry, a fallback, a skipped cell); a run
that met any also says how many when it exits. A run that skipped work
for lack of memory and reported success is how a 64-view benchmark cell
went missing unnoticed.
"""
import atexit
import sys
import traceback

import torch

_GIB = 2 ** 30
# How many exhausted cards this process has announced
_events = 0


def _write(message: str):
    """To the error stream, through tqdm when it is around so that a progress bar is not torn."""
    try:
        from tqdm import tqdm
        tqdm.write(message, file=sys.stderr)
    except ImportError:
        print(message, file=sys.stderr)
    sys.stderr.flush()


def _announce(error: BaseException):
    global _events
    _events += 1
    frames = traceback.extract_tb(error.__traceback__) if error.__traceback__ is not None else []
    where = f"{frames[-1].filename}:{frames[-1].lineno} in {frames[-1].name}" if frames else "unknown place"
    memory = ""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        memory = (
            f": allocated {torch.cuda.memory_allocated() / _GIB:.1f} GiB, reserved "
            f"{torch.cuda.memory_reserved() / _GIB:.1f}, peak {torch.cuda.max_memory_allocated() / _GIB:.1f} "
            f"of {total / _GIB:.1f}"
        )
    first_line = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    _write(f"!! CUDA OUT OF MEMORY #{_events} at {where}{memory} | {first_line[:200]}")


@atexit.register
def _summary():
    if _events:
        _write(f"!! This run ran out of GPU memory {_events} time(s); see the '!! CUDA OUT OF MEMORY' lines above")


def out_of_memory(error: BaseException, announce: bool = True) -> bool:
    """
    Whether an exception is the card running out of memory: either
    torch's OutOfMemoryError, or a RuntimeError carrying one of the
    wordings spconv, cuDNN and cuBLAS report an exhausted card with.
    One that is gets announced on the error stream unless 'announce'
    is off; call this before dropping the error's traceback, which is
    where the announcement reads the place it was raised from.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        exhausted = True
    elif not isinstance(error, RuntimeError):
        exhausted = False
    else:
        message = str(error).lower()
        exhausted = any(
            wording in message for wording in (
                "out of memory",
                "unable to find an engine to execute this computation",
                "cublas_status_alloc_failed",
            )
        )
    if exhausted and announce:
        _announce(error)
    return exhausted


def out_of_memory_events() -> int:
    """How many exhausted cards this process has announced so far."""
    return _events


__all__ = [
    "out_of_memory",
    "out_of_memory_events",
]
