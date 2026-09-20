"""
Score every pruning method of the protocol on the DL3DV test split.

Every test scene is a window of contiguous frames, alternating context
and test frames. At each context view count, each generator predicts a
field from the first frames of the window, and every method cuts that
field down to each budget: a thinning rule, with or without opacity
compensation, with or without a SplatFormer refiner. The whole field
and each pruned one are rendered onto the context views
(self-reconstruction) and onto the test views between them (novel view
synthesis), and scored on PSNR, averaged over the views and then over
the scenes.

The test split is the one training holds out (splits in base.yaml).
The frozen generators were trained on DL3DV by their authors, which is
out of our hands.
"""
import gc
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch import Generator, Tensor
from tqdm import tqdm

from anyprune.datasets import DL3DVDataset, split_scenes
from anyprune.evaluation import psnr
from anyprune.gaussians import Gaussians, Pruner, blending_weights
from anyprune.models import RECONSTRUCTORS, SplatFormer, build_reconstructor
from anyprune.training import Reconstruction, reconstruct, sample_view_indices
from anyprune.utils import load_dotenv, out_of_memory, set_rng_seed
from anyprune.viz import REFERENCE_COLOR, SERIES_COLORS, plot_eval_histogram


# The two halves of a scene, and the views of the reconstruction they are.
BLOCKS = (("self", "context"), ("nvs", "test"))

# What a cell of the summary averages over the scenes.
AVERAGED = ("num_gaussians", "predicted_gaussians", "self_psnr", "nvs_psnr")


@dataclass
class Method:
    """
    One bar of the figure: the rule that picks the Gaussians, or none
    for the whole field, and what is done to the ones it keeps.
    """
    name: str
    rule: Optional[str] = None
    compensate: bool = False
    checkpoint: Optional[str] = None


def score(
    gaussians: Gaussians, reconstruction: Reconstruction, max_intersections: int
) -> Dict[str, float]:
    """Mean PSNR of a field on each half of the scene."""
    scores = {}
    for block, half in BLOCKS:
        views = getattr(reconstruction, half)
        rendered, _ = gaussians.rasterize(
            views.poses, views.intrinsics, views.image_shape,
            max_intersections=max_intersections,
        )
        scores[f"{block}_psnr"] = psnr(rendered, views.images).mean().item()
    return scores


@torch.no_grad()
def evaluate(
    cfg,
    reconstructor,
    refiners: Mapping[str, SplatFormer],
    pruners: Mapping[str, Pruner],
    methods: Sequence[Method],
    scene: Dict[str, Tensor],
    context_idx: Tensor,
    test_idx: Tensor,
    seed: int,
) -> List[dict]:
    """
    Every method at every budget on one scene at one view count. Each
    rule ranks the whole field once and every budget keeps a prefix of
    that order.
    """
    reconstruction = reconstruct(reconstructor, scene, context_idx, test_idx)
    field, context = reconstruction.gaussians, reconstruction.context
    predicted = field.num_gaussians
    measured = None
    if any(pruner.measured for pruner in pruners.values()):
        measured = blending_weights(
            field, context.poses, context.intrinsics, context.image_shape,
            batches_per_pass=cfg.batches_per_pass,
            max_intersections=cfg.device_max_intersections,
        )
    orders = {
        name: pruner.order(
            field, context.poses, context.intrinsics, context.image_shape,
            generator=Generator().manual_seed(seed), measured=measured,
        )
        for name, pruner in pruners.items()
    }
    del measured
    full = score(field, reconstruction, cfg.device_max_intersections)

    records = []
    for budget in cfg.gaussian_budgets:
        for method in methods:
            if method.rule is None:
                kept, scored = predicted, full
            else:
                pruned = field[orders[method.rule][:budget]]
                kept = pruned.num_gaussians
                if method.compensate:
                    pruned = pruned.compensate(
                        kept / predicted, exponent=cfg.compensation.exponent
                    )
                if method.checkpoint is not None:
                    # spconv has no half-precision kernels out of training mode
                    with torch.cuda.amp.autocast(enabled=False):
                        pruned = refiners[method.name](pruned)
                scored = score(pruned, reconstruction, cfg.device_max_intersections)
                del pruned
            records.append({
                "context_views": len(context_idx),
                "method": method.name,
                "budget": budget,
                "num_gaussians": kept,
                "predicted_gaussians": predicted,
                **scored,
            })
    return records


def summarize(records: Sequence[dict]) -> List[dict]:
    """The mean over the scenes of every cell."""
    groups = defaultdict(list)
    for record in records:
        key = (record["generator"], record["context_views"], record["method"], record["budget"])
        groups[key].append(record)
    return [
        {
            "generator": generator,
            "context_views": views,
            "method": method,
            "budget": budget,
            "num_scenes": len(group),
            **{key: sum(r[key] for r in group) / len(group) for key in AVERAGED},
        }
        for (generator, views, method, budget), group in groups.items()
    ]


def write_results(path: Path, cfg, scenes: Sequence[str], records: Sequence[dict]) -> None:
    path.write_text(json.dumps({
        "config": OmegaConf.to_container(cfg, resolve=True),
        "scenes": list(scenes),
        "records": list(records),
        "summary": summarize(records),
    }, indent=2))


def tables(
    summary: Sequence[dict],
    generators: Sequence[str],
    context_views: Sequence[int],
    budgets: Sequence[int],
    methods: Sequence[Method],
) -> str:
    """One table per generator and view count: a row per method, a column per budget."""
    cells = {(r["generator"], r["context_views"], r["method"], r["budget"]): r for r in summary}
    blocks = []
    for generator in generators:
        for views in context_views:
            panel = [r for r in summary if r["generator"] == generator and r["context_views"] == views]
            if not panel:
                continue
            headers = ["method"]
            for budget in budgets:
                kept = min(r["num_gaussians"] for r in panel if r["budget"] == budget)
                headers.append(f"{budget // 1000}k" + (f" ({kept:,.0f})" if kept < budget else ""))
            rows = [
                [method.name] + [
                    "-" if (generator, views, method.name, budget) not in cells else
                    "{self_psnr:.2f} / {nvs_psnr:.2f}".format(**cells[(generator, views, method.name, budget)])
                    for budget in budgets
                ]
                for method in methods
            ]
            blocks.append(
                f"\n{generator}, {views} context views "
                f"({panel[0]['predicted_gaussians']:,.0f} Gaussians predicted), "
                f"{panel[0]['num_scenes']} scenes, PSNR self-reconstruction / novel views\n\n"
                + tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)
            )
    return "\n".join(blocks)


def styles(methods: Sequence[Method]) -> List[Tuple[str, str, Optional[str]]]:
    """
    A hue per (rule, compensation), hatched when refined, and a neutral
    bar for the whole field.
    """
    hues: Dict[Tuple[str, bool], str] = {}
    styled = []
    for method in methods:
        if method.rule is None:
            styled.append((method.name, REFERENCE_COLOR, None))
            continue
        color = hues.setdefault(
            (method.rule, method.compensate), SERIES_COLORS[len(hues) % len(SERIES_COLORS)]
        )
        styled.append((method.name, color, "//" if method.checkpoint else None))
    return styled


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    unknown = set(cfg.generators) - set(RECONSTRUCTORS)
    assert not unknown, f"Every generator has to be one of {RECONSTRUCTORS}: {sorted(unknown)}"
    pruners = {
        name: Pruner(name, **OmegaConf.to_container(rule, resolve=True))
        for name, rule in cfg.rules.items()
    }
    methods = [Method(**OmegaConf.to_container(method, resolve=True)) for method in cfg.methods]
    names = [method.name for method in methods]
    assert len(set(names)) == len(names), f"Two methods share a name: {names}"
    for method in methods:
        assert method.rule is None or method.rule in pruners, (
            f"{method.name!r} prunes with {method.rule!r}, which is not one of {list(pruners)}"
        )
    for method in list(methods):
        if method.checkpoint is not None and not Path(method.checkpoint).exists():
            print(f"Skipping {method.name!r}: no checkpoint at {method.checkpoint}")
            methods.remove(method)
    pruners = {
        name: pruner for name, pruner in pruners.items()
        if any(method.rule == name for method in methods)
    }

    load_dotenv()
    set_rng_seed(cfg.seed, deterministic=cfg.deterministic)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    dataset = DL3DVDataset(cfg.dl3dv_root_dir, cfg.dl3dv_images_subdir)
    scenes = split_scenes(len(dataset), cfg.splits, cfg.split_seed)["test"]
    if cfg.num_scenes is not None:
        scenes = scenes[:cfg.num_scenes]
    scene_names = [Path(dataset.scenes[scene_idx]).name for scene_idx in scenes]
    window = 2 * max(cfg.context_views)
    print(
        f"Evaluating {len(scenes)} scenes of the DL3DV test split on windows of "
        f"{window} frames, from {list(cfg.context_views)} context views at "
        + ", ".join(f"{budget // 1000}k" for budget in cfg.gaussian_budgets)
        + " Gaussians, through " + ", ".join(cfg.generators) + ", with "
        + ", ".join(repr(method.name) for method in methods) + "."
    )

    refiners = {
        method.name: SplatFormer(
            method.checkpoint, quiet=True, batch_statistics=cfg.refiner.batch_statistics
        ).to(device).eval()
        for method in methods if method.checkpoint is not None
    }
    output_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {output_dir}")

    records = []
    for generator in cfg.generators:
        reconstructor = build_reconstructor(
            generator, cfg.anysplat_checkpoint, cfg.yonosplat_checkpoint
        ).to(device)
        for scene_idx in tqdm(scenes, desc=generator, unit="scene"):
            seed = cfg.seed + scene_idx
            frames, context_idx, test_idx = sample_view_indices(
                dataset.num_frames(scene_idx), window,
                generator=Generator().manual_seed(seed),
            )
            scene = {
                name: value.to(device)
                for name, value in dataset.get_frames(scene_idx, frames).items()
            }
            for views in cfg.context_views:
                try:
                    measured = evaluate(
                        cfg, reconstructor, refiners, pruners, methods, scene,
                        context_idx[:views], test_idx[:views], seed,
                    )
                except RuntimeError as error:
                    if not out_of_memory(error):
                        raise
                    error.__traceback__ = None
                    measured = []
                    tqdm.write(
                        f"  scene {scene_idx} did not fit at {views} context views "
                        f"through {generator}, skipped"
                    )
                torch.cuda.empty_cache()
                records += [
                    {"scene": scene_idx, "generator": generator, **record}
                    for record in measured
                ]
            del scene
            write_results(output_dir / "results.json", cfg, scene_names, records)
        del reconstructor
        gc.collect()
        torch.cuda.empty_cache()

    summary = summarize(records)
    print(tables(summary, cfg.generators, cfg.context_views, cfg.gaussian_budgets, methods))
    figure = plot_eval_histogram(
        summary, list(cfg.generators), list(cfg.context_views),
        list(cfg.gaussian_budgets), styles(methods),
        title=f"DL3DV test split, {len(scenes)} scenes, PSNR (dB)",
    )
    path = output_dir / "eval_histogram.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
