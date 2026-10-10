"""
Convert test scenes of RealEstate10K / ACID in pixelSplat's .torch chunk format (lhmd/re10k_torch,
lhmd/acid_torch on the Hugging Face hub) to the DL3DV layout that anyprune.datasets.DL3DVDataset reads.

    python scripts/torch_chunks_to_dl3dv.py CHUNK_DIR OUT_DIR --train-index train_index.json
        [--stride 13] [--min-frames 196] [--size 480 270] [--max-scenes N]

- Only chunks of the test split are read (CHUNK_DIR/test/*.torch), and every scene key is checked against
  the training split's index.json: a key found there aborts the conversion.
- Every 'stride'-th frame is kept. These are 30 fps videos that move ~10x less per frame than DL3DV, so
  13 puts 8 context views (16 alternating frames) across 196 source frames, ReSplat's multi-view RE10K
  setting (re10k_start_0_distance_200_ctx_8v_tgt_8v), and 2 context views 26 frames apart.
- Scenes with fewer than 'min-frames' source frames are skipped, so every view count sees the same scenes.
- Poses: the chunks hold OpenCV world-to-camera matrices; transforms.json gets OpenGL camera-to-world.
  Intrinsics: normalized in the chunks, written in pixels of the stored images (pixel-centre convention).
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("chunks", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--train-index", type=Path, required=True)
    p.add_argument("--stride", type=int, default=13)
    p.add_argument("--min-frames", type=int, default=196)
    p.add_argument("--size", type=int, nargs=2, default=(480, 270))
    p.add_argument("--images-subdir", default="images_270")
    p.add_argument("--max-scenes", type=int, default=None)
    a = p.parse_args()
    train = set(json.loads(a.train_index.read_text()))
    w, h = a.size
    written = skipped = 0
    a.out.mkdir(parents=True, exist_ok=True)
    for chunk in sorted((a.chunks / "test").glob("*.torch")):
        for scene in torch.load(chunk, weights_only=True):
            key = scene["key"]
            assert key not in train, f"{key} of {chunk} is in the training split"
            cameras = scene["cameras"].numpy().astype(np.float64)
            if len(cameras) < a.min_frames:
                skipped += 1
                continue
            if a.max_scenes is not None and written >= a.max_scenes:
                continue
            keep = list(range(0, len(cameras), a.stride))
            fx, fy, cx, cy = cameras[0, :4]
            directory = a.out / key
            (directory / a.images_subdir).mkdir(parents=True, exist_ok=True)
            frames = []
            for i, index in enumerate(keep):
                w2c = np.eye(4)
                w2c[:3] = cameras[index, 6:].reshape(3, 4)
                c2w = np.linalg.inv(w2c) @ np.diag([1.0, -1.0, -1.0, 1.0])
                name = f"frame_{i + 1:05d}.png"
                image = Image.open(io.BytesIO(scene["images"][index].numpy().tobytes())).convert("RGB")
                image.resize((w, h), Image.BICUBIC).save(directory / a.images_subdir / name)
                frames.append({"file_path": f"images/{name}", "transform_matrix": c2w.tolist(),
                               "source_frame": index, "timestamp": int(scene["timestamps"][index])})
            (directory / "transforms.json").write_text(json.dumps({
                "w": w, "h": h, "fl_x": fx * w, "fl_y": fy * h, "cx": cx * w - 0.5, "cy": cy * h - 0.5,
                "camera_model": "PINHOLE", "source": {"chunk": chunk.name, "url": scene.get("url"), "key": key,
                                                     "split": "test", "stride": a.stride,
                                                     "source_frames": len(cameras)},
                "frames": frames}, indent=1))
            written += 1
    print(f"wrote {written} scenes to {a.out}; {skipped} shorter than {a.min_frames} frames skipped")


if __name__ == "__main__":
    main()
