"""
How much of a field can be pruned without losing quality: the largest
cut whose render stays within a tolerance of the unpruned one, found
per scene by a search over the count rather than learned.

The constraint is the one a deployment can actually check - the context
views, which are the only images at hand when the field is pruned - and
every method is asked the same question: ranked by your score, how few
Gaussians can you keep while the context-view PSNR stays within
--tolerance of the reference render? The answer is a count per scene,
and the metrics it reaches on both halves of the scene.

    learned mask       the refined field, cut at a threshold on the
                       mask head's logits (its ranking, not its count)
    Speedy-Splat       the predicted field, top-k of the sensitivity,
                       with the opacity compensation
    RadSplat           the same by the peak blending weight

Each is measured against two references: the whole prediction ('input',
the strict reading of lossless, which asks the pipeline never to lose
against the reconstructor) and, for the learned mask, the refined field
left whole ('refined', which asks only that the cut be harmless).

    python scripts/eval_lossless_search.py --tolerance 0.1

Results go under outputs/eval-lossless-search/<run>/<timestamp>/.
"""
import argparse
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
from tabulate import tabulate
from torch import Generator, Tensor
from tqdm import tqdm

import eval_learned as E
from anyprune.datasets import DL3DVDataset, split_scenes
from anyprune.evaluation import psnr
from anyprune.gaussians import Gaussians
from anyprune.models import build_reconstructor
from anyprune.training import reconstruct, sample_view_indices
from anyprune.utils import load_dotenv, out_of_memory, set_rng_seed

SEARCH_STEPS = 8


@torch.no_grad()
def context_psnr(gaussians: Gaussians, context) -> float:
    rendered, _ = gaussians.rasterize(
        context.poses, context.intrinsics, context.image_shape,
        views_per_pass=E.DEVICE_MAX_VIEWS_PER_RENDER, max_intersections=E.DEVICE_MAX_INTERSECTIONS,
    )
    return psnr(rendered, context.images).mean().item()


@torch.no_grad()
def largest_cut(
    cut: Callable[[int], Gaussians], total: int, context, floor_db: float, steps: int = SEARCH_STEPS,
) -> int:
    """
    The smallest count whose render stays at or above 'floor_db' on the
    context views, by bisection on the count: the quality is monotone
    enough in it for a search, and a count that fails is a lower bound
    for every count below it only in that sense - the bisection returns
    the smallest count it *proved* good, which is conservative.
    """
    low, high, best = 1, total, total
    for _ in range(steps):
        if low >= high:
            break
        middle = (low + high) // 2
        if context_psnr(cut(middle), context) >= floor_db:
            best, high = middle, middle - 1
        else:
            low = middle + 1
    return best


@torch.no_grad()
def evaluate_scene(mask: E.LearnedMask, scorer, reconstructor, scene, context_idx, test_idx, tolerance: float) -> List[dict]:
    reconstruction = reconstruct(reconstructor, scene, context_idx, test_idx)
    field, context = reconstruction.gaussians, reconstruction.context
    predicted = field.num_gaussians
    scores = E.measure_scores(field, context, {"speedy-splat", "radsplat"})
    whole_db = context_psnr(field, context)

    handed, handed_score = field, scores[mask.score]
    if predicted > E.DEVICE_MAX_GAUSSIANS:
        top = torch.topk(handed_score, E.DEVICE_MAX_GAUSSIANS, sorted=False).indices
        handed, handed_score = field[top], handed_score[top]
    pruned = E.refine_and_prune(mask.model, E.build_rule(False), handed, handed_score)
    refined = pruned.refined
    logits = pruned.soft["mask"].float()
    refined_db = context_psnr(refined, context)

    records = [
        {"method": "full field", "num_gaussians": predicted, "reference": "-",
         "context_psnr": whole_db, **scorer(field, reconstruction)},
        {"method": "refined field", "num_gaussians": refined.num_gaussians, "reference": "-",
         "context_psnr": refined_db, **scorer(refined, reconstruction)},
        {"method": "learned mask (its own count)", "num_gaussians": pruned.kept.num_gaussians,
         "reference": "-", "context_psnr": context_psnr(pruned.kept, context),
         **scorer(pruned.kept, reconstruction)},
    ]

    order = torch.argsort(logits, descending=True)
    def mask_cut(count: int) -> Gaussians:
        return refined[order[:count]]
    for reference, floor in (("input", whole_db - tolerance), ("refined", refined_db - tolerance)):
        count = largest_cut(mask_cut, refined.num_gaussians, context, floor)
        kept = mask_cut(count)
        records.append({
            "method": "learned mask, searched", "reference": reference, "num_gaussians": count,
            "context_psnr": context_psnr(kept, context), **scorer(kept, reconstruction),
        })
        del kept

    for name in ("speedy-splat", "radsplat"):
        ranking = torch.argsort(scores[name], descending=True)
        def score_cut(count: int, ranking=ranking) -> Gaussians:
            return E.compensated(field[ranking[:count]], predicted)
        count = largest_cut(score_cut, predicted, context, whole_db - tolerance)
        kept = score_cut(count)
        records.append({
            "method": f"{E.SCORES[name][0]} top-k + compensation, searched", "reference": "input",
            "num_gaussians": count, "context_psnr": context_psnr(kept, context),
            **scorer(kept, reconstruction),
        })
        del kept

    del field, refined, scores, pruned
    return [{"context_views": len(context_idx), "predicted_gaussians": predicted, **r} for r in records]


def summarize(records: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["context_views"], r["method"], r["reference"])].append(r)
    keys = ("num_gaussians", "predicted_gaussians", "context_psnr") + tuple(
        f"{b}_{m}" for b, _ in E.BLOCKS for m, _ in E.METRICS
    )
    return [
        {"context_views": v, "method": m, "reference": ref, "num_scenes": len(g),
         **{k: sum(r[k] for r in g) / len(g) for k in keys}}
        for (v, m, ref), g in groups.items()
    ]


def table(summary: List[dict], views: int) -> str:
    rows = [r for r in summary if r["context_views"] == views]
    body = [[
        r["method"], r["reference"], f"{r['num_gaussians']:,.0f}",
        f"{r['num_gaussians'] / r['predicted_gaussians']:.1%}", f"{r['context_psnr']:.2f}",
        f"{r['nvs_psnr']:.2f}", f"{r['nvs_ssim']:.3f}", f"{r['nvs_lpips']:.3f}",
        f"{r['self_psnr']:.2f}", f"{r['self_ssim']:.3f}", f"{r['self_lpips']:.3f}",
    ] for r in rows]
    return tabulate(body, headers=["method", "vs", "kept", "share", "context PSNR", "nvs PSNR",
                                   "nvs SSIM", "nvs LPIPS", "self PSNR", "self SSIM", "self LPIPS"],
                    tablefmt="simple", disable_numparse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask", default=f"Speedy={E.LEARNED_CHECKPOINT},speedy-splat")
    parser.add_argument("--tolerance", type=float, default=0.1, help="dB of context-view PSNR the cut may cost")
    parser.add_argument("--num-scenes", type=int, default=None)
    parser.add_argument("--context-views", type=int, nargs="+", default=[2, 4, 8, 16, 24])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    load_dotenv(); set_rng_seed(E.SEED, deterministic=False)
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
    run_name = args.run_name or f"tolerance-{args.tolerance:g}"
    output_dir = E.OUTPUTS_DIR / "eval-lossless-search" / run_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    window = 2 * max(args.context_views)
    print(f"Searching the largest cut within {args.tolerance} dB on {len(test_scenes)} scenes, "
          f"{args.context_views} context views. Writing to {output_dir}")

    records, done = [], []
    for views in sorted(args.context_views):
        fitted = 0
        for scene_idx in tqdm(test_scenes, desc=f"{views} context views", unit="scene"):
            frames, context_idx, test_idx = sample_view_indices(
                dataset.num_frames(scene_idx), window, generator=Generator().manual_seed(E.SEED + scene_idx))
            scene = {k: v.to(device) for k, v in dataset.get_frames(scene_idx, frames).items()}
            try:
                measured = evaluate_scene(mask, scorer, reconstructor, scene, context_idx[:views], test_idx[:views], args.tolerance)
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
            (output_dir / "results.json").write_text(json.dumps(
                {"tolerance": args.tolerance, "mask": str(mask.checkpoint), "records": records,
                 "summary": summarize(records)}, indent=2))
        if fitted == 0:
            break
        done.append(views)
    summary = summarize(records)
    report = [f"The largest cut within {args.tolerance} dB of the reference on the context views, "
              f"{len(test_scenes)} DL3DV test scenes"]
    for views in done:
        report += [f"\n{views} context views\n", table(summary, views)]
    report = "\n".join(report)
    print(report)
    (output_dir / "summary.txt").write_text(report)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
