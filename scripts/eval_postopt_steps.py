"""
How many steps of per-scene optimization a pruned field needs: the
whole prediction, the learned mask as refined, and the count-matched
top-k of the mask's score, each optimized once on the context views
(anyprune.gaussians.optimization) with PSNR, SSIM and LPIPS on both
halves of the scene recorded after each of --stages steps, on the DL3DV
test split, the way scripts/eval_learned.py samples it.

    python scripts/eval_postopt_steps.py --stages 10 25 50 100 200 300 500

Results go under outputs/eval-postopt-steps/<run>/<timestamp>/:
results.json, summary.txt (one table per view count, method x steps)
and postopt_steps.png (novel-view metrics against the steps).
"""
import argparse
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tabulate import tabulate
from torch import Generator
from tqdm import tqdm

import eval_learned as E
from anyprune.datasets import DL3DVDataset, split_scenes
from anyprune.gaussians import fine_tune_stages
from anyprune.models import build_reconstructor
from anyprune.training import reconstruct, sample_view_indices
from anyprune.utils import load_dotenv, out_of_memory, set_rng_seed
from anyprune.viz._common import AXIS_COLOR, INK

STAGES = [10, 25, 50, 100, 200, 300, 500]
OUTPUT_ROOT = E.OUTPUTS_DIR / "eval-postopt-steps"


def evaluate_scene(mask: E.LearnedMask, stages: Sequence[int], scorer, reconstructor, scene, context_idx, test_idx, scene_idx) -> List[dict]:
    with torch.no_grad():
        reconstruction = reconstruct(reconstructor, scene, context_idx, test_idx)
        field, context = reconstruction.gaussians, reconstruction.context
        predicted = field.num_gaussians
        score = E.measure_scores(field, context, {mask.score})[mask.score]
        handed, handed_score = field, score
        if predicted > E.DEVICE_MAX_GAUSSIANS:
            top = torch.topk(score, E.DEVICE_MAX_GAUSSIANS, sorted=False).indices
            handed, handed_score = field[top], score[top]
        pruned = E.refine_and_prune(mask.model, E.build_rule(False), handed, handed_score)
        survivors = pruned.kept
        count = survivors.num_gaussians
        matched = E.compensated(field[torch.topk(score, count, sorted=False).indices], predicted)
        del pruned, handed, handed_score, score
    fields = [(E.METHOD_FULL, field), (mask.method, survivors), (E.method_topk(mask.score), matched)]
    records = []
    for name, gaussians in fields:
        records.append({"method": name, "steps": 0, "num_gaussians": gaussians.num_gaussians, **scorer(gaussians, reconstruction)})
        for steps, tuned in fine_tune_stages(
            gaussians, context.poses, context.intrinsics, context.images, stages,
            generator=Generator().manual_seed(E.SEED + scene_idx),
        ):
            records.append({"method": name, "steps": steps, "num_gaussians": tuned.num_gaussians, **scorer(tuned, reconstruction)})
            del tuned
    return [{"context_views": len(context_idx), "predicted_gaussians": predicted, **r} for r in records]


def summarize(records: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["context_views"], r["method"], r["steps"])].append(r)
    return [
        {"context_views": views, "method": method, "steps": steps, "num_scenes": len(group),
         **{key: sum(r[key] for r in group) / len(group) for key in E.AVERAGED}}
        for (views, method, steps), group in groups.items()
    ]


def table(summary: List[dict], views: int) -> str:
    rows = [r for r in summary if r["context_views"] == views]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    steps = sorted(set(r["steps"] for r in rows))
    lines = []
    for block, _ in E.BLOCKS:
        for metric, label in E.METRICS:
            lines.append(f"\n{block} {label}")
            body = []
            for method in methods:
                cells = {r["steps"]: r for r in rows if r["method"] == method}
                body.append([method, f"{cells[0]['num_gaussians'] / cells[0]['predicted_gaussians']:.1%}"] + [
                    (f"{cells[s][f'{block}_{metric}']:.2f}" if metric == "psnr" else f"{cells[s][f'{block}_{metric}']:.3f}")
                    if s in cells else "-" for s in steps
                ])
            lines.append(tabulate(body, headers=["method", "share"] + [f"{s} steps" for s in steps], tablefmt="simple", disable_numparse=True))
    return "\n".join(lines)


def plot(summary: List[dict], context_views: Sequence[int], masks: Sequence[E.LearnedMask], path: Path):
    figure, axes = plt.subplots(3, len(context_views), figsize=(3.6 * len(context_views), 8.5), squeeze=False)
    for column, views in enumerate(context_views):
        rows = [r for r in summary if r["context_views"] == views]
        methods = list(dict.fromkeys(r["method"] for r in rows))
        for line, (metric, label) in enumerate(E.METRICS):
            axis = axes[line, column]
            for method in methods:
                cells = sorted((r["steps"], r[f"nvs_{metric}"]) for r in rows if r["method"] == method)
                _, color, _ = E.style_of(method, masks)
                axis.plot([c[0] for c in cells], [c[1] for c in cells], marker="o", markersize=3, color=color, label=method, linewidth=1.2)
            axis.set_xscale("symlog", linthresh=10)
            axis.set_ylabel(f"novel views {label}", fontsize=8, color=INK)
            axis.set_xlabel("optimization steps", fontsize=8, color=INK)
            if line == 0:
                axis.set_title(f"{views} context views", fontsize=10, color=INK, loc="left")
            axis.tick_params(labelsize=7, colors=INK, length=0)
            axis.grid(True, color=AXIS_COLOR, linewidth=0.6, alpha=0.5)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask", default=f"Speedy prior={E.LEARNED_CHECKPOINT},speedy-splat", metavar="LABEL=PATH[,SCORE[,REFERENCE]]")
    parser.add_argument("--stages", type=int, nargs="+", default=STAGES)
    parser.add_argument("--num-scenes", type=int, default=None)
    parser.add_argument("--context-views", type=int, nargs="+", default=[2, 4, 8, 16, 24])
    parser.add_argument("--run-name", default="postopt-steps")
    args = parser.parse_args()

    load_dotenv()
    set_rng_seed(E.SEED, deterministic=False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    mask = E.parse_mask(args.mask)
    mask.model = E.build_refiner(mask.checkpoint, device, mask_head=True, importance=False, reference=mask.reference)
    dataset = DL3DVDataset(str(E.DL3DV_ROOT_DIR), E.DL3DV_IMAGES_SUBDIR)
    test_scenes = split_scenes(len(dataset), E.SPLITS, E.SPLIT_SEED)["test"]
    if args.num_scenes is not None:
        test_scenes = test_scenes[:args.num_scenes]
    reconstructor = build_reconstructor(E.RECONSTRUCTOR, E.ANYSPLAT_CHECKPOINT, E.YONOSPLAT_CHECKPOINT).to(device)
    scorer = E.Scorer()
    output_dir = OUTPUT_ROOT / args.run_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    window = 2 * max(args.context_views)
    print(f"Optimizing on {len(test_scenes)} test scenes, {args.context_views} context views, stages {args.stages}. Writing to {output_dir}")

    records, done = [], []
    for views in sorted(args.context_views):
        fitted = 0
        for scene_idx in tqdm(test_scenes, desc=f"{views} context views", unit="scene"):
            frames, context_idx, test_idx = sample_view_indices(
                dataset.num_frames(scene_idx), window, generator=Generator().manual_seed(E.SEED + scene_idx),
            )
            scene = {k: v.to(device) for k, v in dataset.get_frames(scene_idx, frames).items()}
            try:
                measured = evaluate_scene(mask, args.stages, scorer, reconstructor, scene, context_idx[:views], test_idx[:views], scene_idx)
            except RuntimeError as error:
                if not out_of_memory(error):
                    raise
                error.__traceback__ = None
                measured = []
                tqdm.write(f"  scene {scene_idx} did not fit at {views} context views, skipped")
            del scene
            torch.cuda.empty_cache()
            fitted += bool(measured)
            records += [{"scene": scene_idx, **r} for r in measured]
            (output_dir / "results.json").write_text(json.dumps({
                "mask": {"label": mask.label, "checkpoint": str(mask.checkpoint), "score": mask.score},
                "stages": args.stages, "records": records, "summary": summarize(records),
            }, indent=2))
        if fitted == 0:
            break
        done.append(views)
    summary = summarize(records)
    report = [f"Per-scene optimization steps, {len(test_scenes)} DL3DV test scenes, mask {mask.label!r}"]
    for views in done:
        report += [f"\n\n{views} context views", table(summary, views)]
    report = "\n".join(report)
    print(report)
    (output_dir / "summary.txt").write_text(report)
    plot(summary, done, [mask], output_dir / "postopt_steps.png")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
