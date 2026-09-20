"""
Score a SplatFormer whose pruning is its own call - the learned mask of
anyprune.training.learned_pruning - against the RadSplat rules at the
count it kept, on the DL3DV test split.

The protocol of scripts/eval.py cuts every method to fixed budgets,
which a learned rule does not take, so here the budget of every scene
is the rule's own: every count-matched control is cut per scene at
what the learned mask kept. Without that, a rule that simply keeps more
primitives would look better for free.

Every test scene is a window of 2 * max(context_views) contiguous
frames, alternating context and test frames; at each view count the
generator predicts a field from the first frames of the window, and
every method is scored on the context views (self-reconstruction) and
on the test views between them (novel view synthesis), on PSNR, SSIM
and LPIPS. The methods, in the order of the figure:

    full field                                  the whole prediction
    learned mask + SplatFormer                  the mask, as refined
    above t=0.01 + compensation                 the RadSplat threshold,
                                                at its own count
    top-k + compensation                        the top of the score at
                                                the learned mask's count
    top-k + compensation + SplatFormer          the same, through a
                                                refiner trained on
                                                compensated top-k fields

and, only when --compensated-checkpoint names one, the mask whose
survivors are compensated by the share kept and which was trained that
way (pruning.learned.compensate). The refiners were trained by
scripts/train.py on fields from 2, 4, 8 and 16 context views, drawn at
random per step, with the loss on every view of the window, for the
same number of steps (see configs/train.yaml), and are read from
checkpoints/anyprune/ unless named on the command line; the view counts
above 16 say how they generalize to fields they were never trained on. A field wider than
the card holds is cut to the top of the score before a refiner sees it,
the way the protocol does. The counts are run from the fewest views up,
each over every scene, and the sweep stops at the first count no scene
fits.

    python scripts/eval_learned.py

Results go under outputs/eval-learned/<run>/<timestamp>/: results.json,
summary.txt, eval_histogram.png (drawn the way scripts/eval.py draws
its protocol, the one 'budget' of every panel being the card's ceiling
and every bar carrying the count its method kept) and
gaussian_counts.png, the count of every test scene: predicted, above
the threshold, and kept by the mask.
"""
import argparse
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from tabulate import tabulate
from torch import Generator, Tensor
from tqdm import tqdm

from anyprune.datasets import DL3DVDataset, split_scenes
from anyprune.evaluation import LPIPS, psnr, ssim
from anyprune.gaussians import Gaussians, blending_weights, radsplat_score
from anyprune.models import SplatFormer, build_reconstructor
from anyprune.training import LearnedRule, Pruned, Reconstruction, ViewSet, apply_rule, reconstruct, sample_view_indices
from anyprune.utils import load_dotenv, out_of_memory, set_rng_seed
from anyprune.viz import REFERENCE_COLOR, SERIES_COLORS, plot_eval_histogram
from anyprune.viz._common import AXIS_COLOR, INK


# --- configs/base.yaml, configs/machine/desktop.yaml ---
SEED = 42
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 0
COMPENSATION_EXPONENT = 1.0
ANYSPLAT_CHECKPOINT = "lhjiang/anysplat"
YONOSPLAT_CHECKPOINT = "botaoye/YoNoSplat/dl3dv_224x224_ctx2to32.ckpt"
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
DL3DV_ROOT_DIR = Path("/home/nicola/Desktop/Anysplat-improved/datasets/dl3dv")
DL3DV_IMAGES_SUBDIR = "images_8"
DEVICE_MAX_GAUSSIANS = 850_000
DEVICE_MAX_VIEWS_PER_RENDER = 2
DEVICE_MAX_INTERSECTIONS = 100_000_000
PRUNING_BATCHES_PER_PASS = 32
RECONSTRUCTOR = "YoNoSplat"
REFINER_BATCH_STATISTICS = True

# --- The learned rule, as configs/train.yaml has it ---
IMPORTANCE_REFERENCE = 0.01
IMPORTANCE_EPS = 1e-6
MASK_WEIGHT = 0.01
MASK_TAU = 0.5
# The threshold of the non-learned baseline, which is also where the
# mask starts from
THRESHOLD = IMPORTANCE_REFERENCE

# --- The refiners, trained by scripts/train.py at 2-16 views with the
# loss on every view of the window (the -2-16v.pth ones without the
# suffix saw at most 8 of them) ---
LEARNED_CHECKPOINT = CHECKPOINTS_DIR / "anyprune" / "learned-mask-2-16v-allviews.pth"
# The compensated mask, checkpoints/anyprune/learned-mask-compensated-2v.pth
# (trained at 2 views only), is scored only when asked for on the
# command line
TOPK_COMPENSATED_CHECKPOINT = CHECKPOINTS_DIR / "anyprune" / "radsplat-topk-compensated-2-16v-allviews.pth"

# --- configs/eval.yaml's counts, with the four the refiners were trained at ---
EVAL_CONTEXT_VIEWS = [2, 4, 8, 16, 24, 32, 64]
EVAL_OUTPUT_ROOT = OUTPUTS_DIR / "eval-learned"
WANDB_PROJECT = "anyprune"


# ---------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------

METHOD_FULL = "full field"
METHOD_LEARNED_COMP = "learned mask + compensation + SplatFormer"
METHOD_LEARNED = "learned mask + SplatFormer"
METHOD_THRESHOLD_COMP = f"above t={THRESHOLD:g} + compensation"
METHOD_TOPK_COMP = "top-k + compensation"
METHOD_TOPK_COMP_REFINED = "top-k + compensation + SplatFormer"
METHODS = [
    METHOD_FULL, METHOD_LEARNED_COMP, METHOD_LEARNED, METHOD_THRESHOLD_COMP,
    METHOD_TOPK_COMP, METHOD_TOPK_COMP_REFINED,
]
# The bars of the figure, as scripts/eval.py styles its own: a hue per
# rule, hatched when refined, and a neutral bar for the whole field
STYLES = [
    (METHOD_FULL, REFERENCE_COLOR, None),
    (METHOD_LEARNED_COMP, SERIES_COLORS[4], "//"),
    (METHOD_LEARNED, SERIES_COLORS[2], "//"),
    (METHOD_THRESHOLD_COMP, SERIES_COLORS[1], None),
    (METHOD_TOPK_COMP, SERIES_COLORS[3], None),
    (METHOD_TOPK_COMP_REFINED, SERIES_COLORS[3], "//"),
]
METRICS = (("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("lpips", "LPIPS"))
BLOCKS = (("self", "context"), ("nvs", "test"))
AVERAGED = ("num_gaussians", "predicted_gaussians") + tuple(
    f"{block}_{metric}" for block, _ in BLOCKS for metric, _ in METRICS
)


def build_rule(compensate: bool) -> LearnedRule:
    return LearnedRule(
        mask=True, mask_weight=MASK_WEIGHT, mask_tau=MASK_TAU,
        compensate=compensate, compensation_exponent=COMPENSATION_EXPONENT,
    )


def build_refiner(checkpoint: Path, device: torch.device, mask_head: bool, importance: bool) -> SplatFormer:
    return SplatFormer(
        str(checkpoint), quiet=True, batch_statistics=REFINER_BATCH_STATISTICS,
        importance_input=importance, mask_head=mask_head,
        importance_reference=IMPORTANCE_REFERENCE, importance_eps=IMPORTANCE_EPS,
    ).to(device).eval()


@torch.no_grad()
def refine_and_prune(model: SplatFormer, rule: LearnedRule, field: Gaussians, score: Tensor) -> Pruned:
    """
    Refine a field and read the mask off the same forward, in single
    precision (spconv has no half-precision kernel out of training
    mode), with no Gumbel noise.
    """
    with torch.cuda.amp.autocast(enabled=False):
        refined, logits = model.refine(field, score)
    return apply_rule(rule, refined, logits, noisy=False)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

class Scorer:
    """PSNR, SSIM and LPIPS of a field on each half of a scene, off one render per half."""

    def __init__(self):
        self.lpips = LPIPS()

    @torch.no_grad()
    def __call__(self, gaussians: Gaussians, reconstruction: Reconstruction) -> Dict[str, float]:
        scores = {}
        for block, half in BLOCKS:
            views: ViewSet = getattr(reconstruction, half)
            rendered, _ = gaussians.rasterize(
                views.poses, views.intrinsics, views.image_shape,
                views_per_pass=DEVICE_MAX_VIEWS_PER_RENDER,
                max_intersections=DEVICE_MAX_INTERSECTIONS,
            )
            scores[f"{block}_psnr"] = psnr(rendered, views.images).mean().item()
            scores[f"{block}_ssim"] = ssim(rendered, views.images).mean().item()
            scores[f"{block}_lpips"] = self.lpips(rendered, views.images).mean().item()
        return scores


def compensated(gaussians: Gaussians, predicted: int) -> Gaussians:
    """A thinned field with the optical depth of the thinning put back."""
    return gaussians.compensate(gaussians.num_gaussians / predicted, exponent=COMPENSATION_EXPONENT)


@torch.no_grad()
def evaluate_scene(
    learned: SplatFormer, learned_comp: Optional[SplatFormer], topk_refiner: Optional[SplatFormer],
    scorer: Scorer, reconstructor, scene, context_idx, test_idx,
) -> List[dict]:
    """
    Every method on one scene at one view count, off a single sweep of
    the field over the context views. The top-k controls are cut at
    exactly the count the learned mask (as refined, without
    compensation) kept; the threshold keeps whatever scores above it.
    """
    reconstruction = reconstruct(reconstructor, scene, context_idx, test_idx)
    field, context = reconstruction.gaussians, reconstruction.context
    predicted = field.num_gaussians
    measured = blending_weights(
        field, context.poses, context.intrinsics, context.image_shape,
        batches_per_pass=PRUNING_BATCHES_PER_PASS, max_intersections=DEVICE_MAX_INTERSECTIONS,
    )
    score = radsplat_score(measured)
    del measured
    above = int((score >= THRESHOLD).sum().item())

    records = [{"method": METHOD_FULL, "num_gaussians": predicted, **scorer(field, reconstruction)}]

    # What the card holds of the prediction, refined and pruned by each
    # learned model
    handed, handed_score = field, score
    if predicted > DEVICE_MAX_GAUSSIANS:
        top = torch.topk(score, DEVICE_MAX_GAUSSIANS, sorted=False).indices
        handed, handed_score = field[top], score[top]
    count = None
    for name, model, compensate in (
        (METHOD_LEARNED_COMP, learned_comp, True), (METHOD_LEARNED, learned, False),
    ):
        if model is None:
            continue
        pruned = refine_and_prune(model, build_rule(compensate), handed, handed_score)
        survivors = pruned.kept
        if not compensate:
            count = survivors.num_gaussians
        records.append({
            "method": name, "num_gaussians": survivors.num_gaussians,
            "input_gaussians": handed.num_gaussians,
            "soft_keep": pruned.soft["mask"].mean().item(),
            "scale_ratio": (pruned.refined.scales.float().mean() / handed.scales.float().mean()).item(),
            **scorer(survivors, reconstruction),
        })
        del pruned, survivors
    del handed

    # The threshold, at its own count
    kept = torch.nonzero(score >= THRESHOLD).squeeze(1)
    if kept.numel() == 0:
        kept = score.argmax().reshape(1)
    thresholded = compensated(field[kept], predicted)
    records.append({
        "method": METHOD_THRESHOLD_COMP, "num_gaussians": thresholded.num_gaussians,
        **scorer(thresholded, reconstruction),
    })
    del thresholded

    # The top of the score at the learned mask's count, compensated,
    # raw and through the refiner trained on such fields
    if count is not None:
        matched = compensated(field[torch.topk(score, count, sorted=False).indices], predicted)
        records.append({
            "method": METHOD_TOPK_COMP, "num_gaussians": matched.num_gaussians,
            **scorer(matched, reconstruction),
        })
        if topk_refiner is not None:
            try:
                with torch.cuda.amp.autocast(enabled=False):
                    refined = topk_refiner(matched)
                records.append({
                    "method": METHOD_TOPK_COMP_REFINED, "num_gaussians": refined.num_gaussians,
                    **scorer(refined, reconstruction),
                })
                del refined
            except RuntimeError as error:
                if not out_of_memory(error):
                    raise
                error.__traceback__ = None
                torch.cuda.empty_cache()
                tqdm.write(f"  {count:,} Gaussians did not fit {METHOD_TOPK_COMP_REFINED}, skipped")
        del matched
    del field, score
    # Keyed the way the protocol's records are, so that its figure draws
    # these: the one budget is the card's ceiling, and what a cell kept
    # under it is the rule's count
    return [
        {
            "generator": RECONSTRUCTOR, "context_views": len(context_idx),
            "budget": DEVICE_MAX_GAUSSIANS, "predicted_gaussians": predicted,
            "above_threshold": above, **record,
        }
        for record in records
    ]


def summarize(records: List[dict]) -> List[dict]:
    """The mean over the scenes of every (view count, method) cell."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["context_views"], record["method"])].append(record)
    return [
        {
            "generator": RECONSTRUCTOR, "context_views": views, "method": method,
            "budget": DEVICE_MAX_GAUSSIANS, "num_scenes": len(group),
            **{key: sum(r[key] for r in group) / len(group) for key in AVERAGED},
        }
        for (views, method), group in groups.items()
    ]


def table(summary: List[dict], views: int) -> str:
    """One row per method: what it kept, and every metric on both halves."""
    cells = {r["method"]: r for r in summary if r["context_views"] == views}
    rows = []
    for method in METHODS:
        r = cells.get(method)
        if r is None:
            continue
        rows.append([
            method, r["num_scenes"], f"{r['num_gaussians']:,.0f}",
            f"{r['num_gaussians'] / r['predicted_gaussians']:.1%}",
        ] + [
            f"{r[f'{block}_{metric}']:.3f}" if metric != "psnr" else f"{r[f'{block}_{metric}']:.2f}"
            for block, _ in BLOCKS for metric, _ in METRICS
        ])
    headers = ["method", "scenes", "kept", "share"] + [
        f"{block} {metric.upper()}" for block, _ in BLOCKS for metric, _ in METRICS
    ]
    return tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)


def plot_counts(records: Sequence[dict], context_views: Sequence[int], title: str) -> Figure:
    """
    The Gaussians of every test scene, at each view count: what the
    generator predicted, what scores above the threshold, and what the
    learned mask kept.
    """
    series = [
        ("predicted", REFERENCE_COLOR, lambda r: r["predicted_gaussians"]),
        (f"above t={THRESHOLD:g}", SERIES_COLORS[1], lambda r: r["above_threshold"]),
        ("kept by the learned mask", SERIES_COLORS[2], lambda r: r["num_gaussians"]),
    ]
    figure, axes = plt.subplots(
        len(context_views), 1, figsize=(11, 2.8 * len(context_views)), squeeze=False,
    )
    width = 0.8 / len(series)
    for axis, views in zip(axes[:, 0], context_views):
        cells = [r for r in records if r["context_views"] == views and r["method"] == METHOD_LEARNED]
        cells.sort(key=lambda r: r["scene"])
        positions = range(len(cells))
        for i, (label, color, read) in enumerate(series):
            offset = (i - (len(series) - 1) / 2) * width
            axis.bar(
                [p + offset for p in positions], [read(r) for r in cells], width,
                color=color, edgecolor="white", linewidth=0.4, label=label,
            )
        axis.set_xticks(list(positions))
        axis.set_xticklabels([str(r["scene"]) for r in cells], fontsize=7)
        axis.set_xlabel("DL3DV test scene", fontsize=8, color=INK)
        axis.set_ylabel("Gaussians", fontsize=9, color=INK)
        axis.set_title(f"{RECONSTRUCTOR}, {views} context views", fontsize=10, color=INK, loc="left")
        axis.tick_params(labelsize=8, colors=INK, length=0)
        axis.grid(True, axis="y", color=AXIS_COLOR, linewidth=0.6, alpha=0.5)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS_COLOR)
    handles = [Patch(facecolor=color, label=label) for label, color, _ in series]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK)
    figure.suptitle(title, fontsize=12, color=INK)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(LEARNED_CHECKPOINT), help="the learned mask refiner")
    parser.add_argument("--compensated-checkpoint", default=None,
                        help="also score the learned mask refiner trained with compensation of what it keeps")
    parser.add_argument("--topk-checkpoint", default=str(TOPK_COMPENSATED_CHECKPOINT),
                        help="the refiner trained on compensated top-k fields")
    parser.add_argument("--importance", action="store_true",
                        help="the learned refiners were trained with the importance input")
    parser.add_argument("--run-name", default=None, help="what to file the results under; the checkpoint's stem by default")
    parser.add_argument("--num-scenes", type=int, default=None, help="how many test scenes, all by default")
    parser.add_argument("--context-views", type=int, nargs="+", default=EVAL_CONTEXT_VIEWS)
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    assert checkpoint.exists(), f"No learned mask refiner at {checkpoint}"
    run_name = args.run_name or checkpoint.stem

    load_dotenv()
    set_rng_seed(SEED, deterministic=False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    dataset = DL3DVDataset(str(DL3DV_ROOT_DIR), DL3DV_IMAGES_SUBDIR)
    test_scenes = split_scenes(len(dataset), SPLITS, SPLIT_SEED)["test"]
    if args.num_scenes is not None:
        test_scenes = test_scenes[:args.num_scenes]

    learned = build_refiner(checkpoint, device, mask_head=True, importance=args.importance)
    learned_comp = topk_refiner = None
    for name, path, mask_head in (
        (METHOD_LEARNED_COMP, args.compensated_checkpoint, True),
        (METHOD_TOPK_COMP_REFINED, args.topk_checkpoint, False),
    ):
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            print(f"Skipping {name!r}: no checkpoint at {path}")
            continue
        refiner = build_refiner(path, device, mask_head=mask_head, importance=args.importance and mask_head)
        if mask_head:
            learned_comp = refiner
        else:
            topk_refiner = refiner
    reconstructor = build_reconstructor(RECONSTRUCTOR, ANYSPLAT_CHECKPOINT, YONOSPLAT_CHECKPOINT).to(device)
    scorer = Scorer()

    output_dir = EVAL_OUTPUT_ROOT / run_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    window = 2 * max(args.context_views)
    checkpoints = {
        METHOD_LEARNED: str(checkpoint),
        METHOD_LEARNED_COMP: args.compensated_checkpoint if learned_comp is not None else None,
        METHOD_TOPK_COMP_REFINED: args.topk_checkpoint if topk_refiner is not None else None,
    }
    print(
        f"Evaluating the learned mask ({checkpoint}) on {len(test_scenes)} scenes of the DL3DV test "
        f"split, windows of {window} frames, from {args.context_views} context views. Writing to {output_dir}"
    )
    use_wandb = args.wandb != "disabled"
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT, mode=args.wandb, name=f"{run_name}-eval", job_type="eval",
            config={"checkpoints": checkpoints, "num_scenes": len(test_scenes),
                    "context_views": args.context_views, "importance_input": args.importance},
            settings=wandb.Settings(x_disable_stats=True),
        )

    def write_results():
        (output_dir / "results.json").write_text(json.dumps({
            "checkpoints": checkpoints, "importance_input": args.importance,
            "scenes": [Path(dataset.scenes[idx]).name for idx in test_scenes],
            "records": records, "summary": summarize(records),
        }, indent=2))

    records, context_views = [], []
    for views in sorted(args.context_views):
        fitted = 0
        for scene_idx in tqdm(test_scenes, desc=f"{views} context views", unit="scene"):
            # The same window of the scene at every count, so that a
            # count's context views are the first of the wider one's
            seed = SEED + scene_idx
            frames, context_idx, test_idx = sample_view_indices(
                dataset.num_frames(scene_idx), window, generator=Generator().manual_seed(seed),
            )
            scene = {
                name: value.to(device)
                for name, value in dataset.get_frames(scene_idx, frames).items()
            }
            try:
                measured = evaluate_scene(
                    learned, learned_comp, topk_refiner, scorer, reconstructor, scene,
                    context_idx[:views], test_idx[:views],
                )
            except RuntimeError as error:
                if not out_of_memory(error):
                    raise
                error.__traceback__ = None
                measured = []
                tqdm.write(f"  scene {scene_idx} did not fit at {views} context views, skipped")
            del scene
            torch.cuda.empty_cache()
            fitted += bool(measured)
            records += [{"scene": scene_idx, **record} for record in measured]
            write_results()
        if fitted == 0:
            print(f"No scene fit at {views} context views: stopping the sweep here")
            break
        context_views.append(views)
    args.context_views = context_views

    summary = summarize(records)
    assert summary, "No scene was measured: nothing to report"
    report = [f"Learned mask ({checkpoint}), {len(test_scenes)} DL3DV test scenes; the top-k controls at its count"]
    for views in args.context_views:
        report += [f"\n{views} context views\n", table(summary, views)]
    report = "\n".join(report)
    print(report)
    (output_dir / "summary.txt").write_text(report)

    drawn = [style for style in STYLES if any(r["method"] == style[0] for r in summary)]
    figure = plot_eval_histogram(
        summary, [RECONSTRUCTOR], args.context_views, [DEVICE_MAX_GAUSSIANS], drawn,
        title=f"DL3DV test split, {len(test_scenes)} scenes, at the count the learned mask kept",
        metrics=METRICS, annotate_counts=True,
    )
    histogram_path = output_dir / "eval_histogram.png"
    figure.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    figure = plot_counts(
        records, args.context_views,
        title=f"Gaussians per test scene: predicted, above the threshold, kept by the learned mask",
    )
    counts_path = output_dir / "gaussian_counts.png"
    figure.savefig(counts_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {histogram_path}, {counts_path} and summary.txt")

    if use_wandb:
        columns = ["context_views", "method", "num_scenes", "predicted_gaussians", "num_gaussians"] + list(AVERAGED[2:])
        rows = [[r["context_views"], r["method"], r["num_scenes"], r["predicted_gaussians"], r["num_gaussians"]]
                + [r[key] for key in AVERAGED[2:]] for r in summary]
        logged = {
            "eval/summary": wandb.Table(columns=columns, data=rows),
            "eval/figure": wandb.Image(str(histogram_path)),
            "eval/gaussian_counts": wandb.Image(str(counts_path)),
        }
        for r in summary:
            slug = "".join(c if c.isalnum() else "-" for c in r["method"].lower()).strip("-")
            for key in AVERAGED:
                logged[f"eval/{r['context_views']}v/{slug}/{key}"] = r[key]
        wandb.log(logged)
        wandb.finish()


if __name__ == "__main__":
    main()
