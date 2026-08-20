import sys
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import random_split, DataLoader

from anyprune.datasets import DL3DVDataset
from anyprune.wrappers import FrozenAnySplat


def _set_rng_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    _set_rng_seed(cfg.seed)

    print("Initializing frozen Anysplat from pre-trained checkpoint...")
    anysplat = FrozenAnySplat(cfg.anysplat_checkpoint, quiet=True)
    print("Done.")

    print("Initializing dataset...")
    dataset = DL3DVDataset(cfg.dl3dv_root_dir, cfg.dl3dv_images_subdir)
    print(f"DL3DV initialized with {len(dataset)} scenes.")

    print(f"{dataset[540]['poses'].shape=}")
    print(f"{dataset[540]['images'].shape=}")


if __name__ == "__main__":
    main()
