"""
Pinning down every source of randomness we go through.
"""
import os
import random

import numpy as np
import torch


def set_rng_seed(seed: int, deterministic: bool = False):
    """
    Seed every generator a run draws from, and optionally ask the
    libraries underneath for reproducible kernels as well.

    Asking for deterministic kernels is only a small change in the 
    numbers, and costs significantly more execution time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


__all__ = [
    "set_rng_seed",
]
