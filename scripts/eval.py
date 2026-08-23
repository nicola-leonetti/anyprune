import sys
import os
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pathlib import Path
from torch import Tensor
from torch.utils.data import random_split, DataLoader

from anyprune import Gaussians
from anyprune.datasets import DL3DVDataset
from anyprune.evaluation import psnr
from anyprune.models import FrozenAnySplat, FrozenYoNoSplat
from anyprune.utils import set_rng_seed, to_reconstruction_frame
from anyprune.viz import ModelReconstruction, plot_reconstructions


SCENE_IDX = 540
# We take a contiguous run of frames, so the views are as dense as the
# capture allows, and alternate them between the context views the
# models reconstruct from and the test views they never see.
FIRST_FRAME = 0
NUM_SAMPLED_VIEWS = 12
# How many of those views the figure shows, spread over the trajectory
NUM_PLOTTED_VIEWS = 5

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "eval.png"


def reconstruct(
    name: str,
    model: nn.Module,
    scene: dict,
    context_idx: Tensor,
    test_idx: Tensor,
) -> ModelReconstruction:
    """
    Reconstruct a scene from its context views with one model, render
    both those views and the held-out ones, and report how close each
    came to the ground truth.
    """
    images = scene["images"]
    image_shape = images.shape[-2:]

    with torch.no_grad():
        pred_poses, pred_intrinsics, gaussians = model(images[context_idx])
        print(f"{name} predicted {gaussians.num_gaussians} Gaussians.")

        # A model hands back the context cameras in its own frame, so
        # those can be rasterized as they are, while the held-out ones
        # come from the dataset and have to be carried over to it first.
        self_recon, _ = gaussians.rasterize(pred_poses, pred_intrinsics, image_shape)
        test_poses = to_reconstruction_frame(
            scene["poses"][context_idx], pred_poses, scene["poses"][test_idx]
        )
        # Use the predicted intrinsics: even though the more accurate
        # ones are the ones from the datasets, they do not agree with
        # the reference frame we are using and cost a lot of PSNR.
        test_intrinsics = pred_intrinsics.mean(dim=0, keepdim=True)
        test_render, _ = gaussians.rasterize(
            test_poses, test_intrinsics.expand(len(test_idx), 3, 3), image_shape
        )

    print(f"{name} self-reconstruction PSNR: {psnr(self_recon, images[context_idx]).mean():.2f} dB")
    print(f"{name} test view PSNR:           {psnr(test_render, images[test_idx]).mean():.2f} dB")

    return ModelReconstruction(
        name=name,
        num_gaussians=gaussians.num_gaussians,
        self_recon=self_recon,
        test_render=test_render,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    set_rng_seed(cfg.seed)
    device = torch.device("cuda")

    # Built one at a time: a feed-forward reconstructor is a few GB of
    # weights, and they do not have to meet to be compared.
    models = {
        "AnySplat": lambda: FrozenAnySplat(cfg.anysplat_checkpoint, quiet=True),
        "YoNoSplat": lambda: FrozenYoNoSplat(cfg.yonosplat_checkpoint, quiet=True),
    }

    print("Initializing dataset...")
    dataset = DL3DVDataset(cfg.dl3dv_root_dir, cfg.dl3dv_images_subdir)
    print(f"DL3DV initialized with {len(dataset)} scenes.")

    scene = dataset[SCENE_IDX]
    scene = {
        "images": scene["images"].to(device),
        "poses": scene["poses"].to(device).reshape(-1, 4, 4),
    }
    images = scene["images"]

    sampled = FIRST_FRAME + torch.arange(NUM_SAMPLED_VIEWS)
    assert sampled[-1] < images.shape[0], (
        f"Scene {SCENE_IDX} only has {images.shape[0]} frames, "
        f"not enough to sample {NUM_SAMPLED_VIEWS} from frame {FIRST_FRAME} on"
    )
    context_idx, test_idx = sampled[::2], sampled[1::2]
    print(f"Using {len(context_idx)} context views and {len(test_idx)} test views.")

    reconstructions = []
    for name, build_model in models.items():
        print(f"Initializing frozen {name} from pre-trained checkpoint...")
        model = build_model().to(device)
        reconstructions.append(reconstruct(name, model, scene, context_idx, test_idx))
        del model
        torch.cuda.empty_cache()

    figure = plot_reconstructions(
        context_images=images[context_idx],
        test_images=images[test_idx],
        reconstructions=reconstructions,
        title=f"DL3DV scene {SCENE_IDX}, reconstructed from {len(context_idx)} context views",
        num_shown=NUM_PLOTTED_VIEWS,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()