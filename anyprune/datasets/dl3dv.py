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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from anyprune import process_image


# Side of the square images process_image() returns.
PROCESSED_IMAGE_SIZE = 448


@lru_cache(maxsize=None)
def _read_transforms_json(scene_dir: str) -> Dict[str, Any]:
    """
    Given the path of a scene directory, parse its transforms.json.
    """
    with open(Path(scene_dir) / "transforms.json") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _read_poses(scene_dir: str) -> Tensor:
    """
    Given the path of a scene directory, return a (V, 1, 4, 4) tensor
    with the poses of all the frames in the scene.
    """
    data = _read_transforms_json(scene_dir)
    poses = [frame["transform_matrix"] for frame in data["frames"]]
    poses = torch.tensor(poses, dtype=torch.float32).unsqueeze(1)
    return poses # (V, 1, 4, 4)


@lru_cache(maxsize=None)
def _read_img_paths(scene_dir: str, img_subdir: str) -> List[str]:
    """
    Given the path of a scene directory and an image directory, return
    an ordered list of the paths of the images.
    """
    data = _read_transforms_json(scene_dir)
    return tuple(
        Path(scene_dir) / img_subdir / Path(frame["file_path"]).name
        for frame in data["frames"]
    )


def _rescale_axis(
    focal: float, 
    principal: float, 
    scale: float
) -> Tuple[float, float]:
    """
    Rescale one axis of a pinhole intrinsic by 'scale'.
    """
    # Pixel centers sit at integer coordinates plus a half
    return focal * scale, (principal + 0.5) * scale - 0.5


@lru_cache(maxsize=None)
def _read_intrinsics(scene_dir: str, img_subdir: str) -> Tensor:
    """
    Given the path of a scene directory and an image directory, return a
    (V, 1, 3, 3) tensor with the pinhole intrinsics of every frame, in
    pixels of the images that _read_image_info() returns.

    DL3DV stores a single camera per scene, calibrated for the original
    full-resolution capture ("w" and "h" in transforms.json). 
    Those intrinsics are then changed with the two transformations the 
    frames go through before we see them:
    - the downscaling to the resolution of the copy on disk in 
        img_subdir.
    - the resize plus center crop in process_image().
    """
    data = _read_transforms_json(scene_dir)
    img_paths = _read_img_paths(scene_dir, img_subdir)

    fx, fy = data["fl_x"], data["fl_y"]
    cx, cy = data["cx"], data["cy"]

    # Original capture resolution -> the downscaled copy on disk
    with Image.open(img_paths[0]) as img:
        disk_w, disk_h = img.size
    fx, cx = _rescale_axis(fx, cx, disk_w / data["w"])
    fy, cy = _rescale_axis(fy, cy, disk_h / data["h"])

    # The copy on disk -> process_image()'s resize to a short edge of
    # PROCESSED_IMAGE_SIZE, followed by a center crop to a square
    if disk_w > disk_h:
        resized_h = PROCESSED_IMAGE_SIZE
        resized_w = int(disk_w * (PROCESSED_IMAGE_SIZE / disk_h))
    else:
        resized_w = PROCESSED_IMAGE_SIZE
        resized_h = int(disk_h * (PROCESSED_IMAGE_SIZE / disk_w))
    fx, cx = _rescale_axis(fx, cx, resized_w / disk_w)
    fy, cy = _rescale_axis(fy, cy, resized_h / disk_h)
    cx -= (resized_w - PROCESSED_IMAGE_SIZE) // 2
    cy -= (resized_h - PROCESSED_IMAGE_SIZE) // 2

    intrinsics = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    return intrinsics.expand(len(img_paths), 1, 3, 3).contiguous() # (V, 1, 3, 3)


def _read_image_info(
    scene_dir: str, img_subdir: str, frames: Optional[Sequence[int]] = None
) -> Tensor:
    """
    Given the directory of a scene and the img_subdir subfolder
    containing the frames, reads the images in the directory to get a
    (V, 3, H, W) tensor with the frames.

    Optionally, 'frames' can be provided to read only a subset of the
    images on disk.
    """
    img_paths = _read_img_paths(scene_dir, img_subdir)
    if frames is not None:
        img_paths = [img_paths[frame] for frame in frames]
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

    def num_frames(self, idx: int) -> int:
        """
        How many frames the scene of index idx has, without reading any
        of them.
        """
        return len(_read_img_paths(self.scenes[idx], self.img_subdir))

    def __getitem__(self, idx):
        return self.get_frames(idx)

    def get_frames(self, idx: int, frames: Optional[Sequence[int]] = None):
        """
        Returns {images: images, poses: poses, intrinsics: intrinsics}

        Images is a (V, 3, H, W) tensor with the V images with indices
        specified in 'frames', or all of the available images if 
        'frames' is left unspecified.

        Images are normalized to be in [0, 1].

        'poses' is a (V, 1, 4, 4) tensor containing the poses of each
        image, as camera-to-world matrices with OpenGL camera axes. See
        Gaussians.rasterize() for the full convention.

        'intrinsics' is a (V, 1, 3, 3) tensor of pinhole camera 
        matrices, expressed in terms of pixels of the corresponding 
        image returned by this method.
        """
        scene_path = self.scenes[idx]
        images = _read_image_info(scene_path, self.img_subdir, frames)
        poses =  _read_poses(scene_path)
        intrinsics = _read_intrinsics(scene_path, self.img_subdir)
        if frames is not None:
            selection = torch.as_tensor(frames, dtype=torch.long)
            poses, intrinsics = poses[selection], intrinsics[selection]
        assert images.shape[-2:] == (PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE), (
            f"Intrinsics are computed for {PROCESSED_IMAGE_SIZE}x{PROCESSED_IMAGE_SIZE} "
            f"images, but the frames are {tuple(images.shape[-2:])}"
        )
        return {
            "images": images,         # (V, 3, H, W)
            "poses": poses,           # (V, 1, 4, 4)
            "intrinsics": intrinsics, # (V, 1, 3, 3)
        }
