"""
Implementation of the DL3DV dataset.

This dataset's directory structure is 
    <root>
    |   <scene_1_hash>
    |   |   <img_subdir>
    |   |   |   <.png image 1>
    |   |   |   <.png image 2>
    |   |   |   ...
    |   |   transforms.json
    |   <scene_2_hash>
    |   ...
"""
import json
import os
from functools import lru_cache
from typing import List, Tuple

import torch
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset

from anyprune import process_image


@lru_cache(maxsize=None)
def _read_poses(scene_dir: str) -> Tensor:
    """
    Given the path of a scene directory, return a (V, 1, 4, 4) tensor
    with the poses of all the frames in the scene.
    """
    with open(Path(scene_dir) / "transforms.json") as f: 
        data = json.load(f)
    poses = [frame["transform_matrix"] for frame in data["frames"]]
    poses = torch.tensor(poses, dtype=torch.float32).unsqueeze(1)
    return poses # (V, 1, 4, 4)


@lru_cache(maxsize=None)
def _read_img_paths(scene_dir: str, img_subdir: str) -> List[str]:
    """
    Given the path of a scene directory and an image directory, return 
    an ordered list of the paths of the images.
    """
    with open(Path(scene_dir) / "transforms.json") as f: 
        data = json.load(f)
    return tuple(
        Path(scene_dir) / img_subdir / Path(frame["file_path"]).name
        for frame in data["frames"]
    )


def _read_image_info(scene_dir: str, img_subdir: str) -> Tensor:
    """
    Given the directory of a scene and the img_subdir subfolder 
    containing the frames, reads the images in the directory to get a 
    (V, 3, H, W) tensor with all the frames.
    """
    img_paths = _read_img_paths(scene_dir, img_subdir)
    images = [process_image(path) for path in img_paths]
    # normalize from [-1, 1] to [0, 1]
    images = torch.stack(images, dim=0).mul_(0.5).add_(0.5)
    return images # (V, 3, H, W)


class DL3DVDataset(Dataset):
    def __init__(self, root: str, img_subdir: str):
        super().__init__()
        self.root = root
        self.img_subdir = img_subdir
        # Cache the absolute path of all scenes
        self.scenes = [
            os.path.join(root, dir_name)
            for dir_name in sorted(os.listdir(root)) 
            if os.path.isdir(os.path.join(root, dir_name))
        ]
            
    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        """
        Returns {images: images, poses: poses}

        Images is a (V, 3, H, W) tensor with all images in the scene of
        index idx, normalized to be in [0, 1].

        Poses is a (V, 1, 4, 4) tensor containing the poses of each 
        image.
        """
        scene_path = self.scenes[idx]
        images = _read_image_info(scene_path, self.img_subdir)
        poses =  _read_poses(scene_path)
        return {
            "images": images, # (V, 3, H, W)
            "poses": poses,   # (V, 1, 4, 4)
        }
         