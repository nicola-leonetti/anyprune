"""
Telling a card that ran out of room apart from a run that went wrong.
"""
import torch


def out_of_memory(error: BaseException) -> bool:
    """
    Whether an exception is the card running out of memory: either
    torch's OutOfMemoryError, or a RuntimeError carrying one of the
    wordings spconv, cuDNN and cuBLAS report an exhausted card with.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return any(
        wording in message for wording in (
            "out of memory",
            "unable to find an engine to execute this computation",
            "cublas_status_alloc_failed",
        )
    )


__all__ = [
    "out_of_memory",
]
