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
and LPIPS. The methods:

    full field                                  the whole prediction
    learned mask (<label>) + SplatFormer        each mask of --masks, as
                                                refined, at its own count
    above t=0.01 + compensation                 the RadSplat threshold,
                                                at its own count
    <score> top-k + compensation                the top of each score of
                                                --baseline-scores (RadSplat,
                                                Speedy-Splat, PUP 3D-GS,
                                                Mini-Splatting, LightGaussian,
                                                REFINE, GaussianPOP) at the
                                                first mask's count
    GaussianPOP N cycles + compensation         GaussianPOP's own procedure
                                                at that count: the cut in
                                                GAUSSIANPOP_CYCLES steps, the
                                                error measured again before
                                                each
    <score> top-k                               the same without the
                                                compensation (--raw-topk)
    <score> own rule (...) [+ compensation]     each score cut where its
                                                paper cuts it (--own-rules):
                                                RadSplat at t=0.01, the others
                                                at the share of the field
                                                their pipelines keep (GaussianPOP's
                                                in its cycles)
    <score> top-k + compensation + SplatFormer  the same, through the
                                                refiner trained on such
                                                fields (--topk-checkpoints)
    ... + post-optimization                     any of the above after
                                                --post-optimize steps of
                                                per-scene 3DGS optimization
                                                on the context views
                                                (anyprune.gaussians.optimization)
    <score> top-k + compensation [at X's count] the raw top of a second
                                                mask's own score at its count

The refiners were trained by scripts/train.py on fields from 2, 4, 8
and 16 context views, drawn at random per step, with the loss on every
view of the window, for the same number of steps (see
configs/train.yaml), and are read from checkpoints/anyprune/ unless
named on the command line; the view counts above 16 say how they
generalize to fields they were never trained on. A field wider than the
card holds is cut to the top of the mask's score before a refiner sees
it, the way the protocol does. The counts are run from the fewest views
up, each over every scene, and the sweep stops at the first count no
scene fits.

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
from dataclasses import dataclass
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
from anyprune.gaussians import (
    GAUSSIANPOP_CYCLES, Gaussians, blending_weights, fine_tune, gaussianpop_prune, gaussianpop_score,
    lightgaussian_score, mini_splatting_score, pup_score, radsplat_score, refine_score, sensitivity_scores,
    speedy_splat_score,
)
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
# Trained with the degradation term at weight 2 reading Speedy-Splat's
# sensitivity (configs/train.yaml), whose reference keeps about the
# same count as RadSplat's 0.01
LEARNED_CHECKPOINT = CHECKPOINTS_DIR / "anyprune" / "learned-mask-2-16v-deg2.0-speedy.pth"
SPEEDY_IMPORTANCE_REFERENCE = 0.0125
# The same recipe reading the RadSplat score (pruning.learned.score),
# 0.2-1.4 dB under it (2026-09-22)
RADSPLAT_LEARNED_CHECKPOINT = CHECKPOINTS_DIR / "anyprune" / "learned-mask-2-16v-deg2.0-input-p2.0.pth"
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
METHOD_THRESHOLD_COMP = f"above t={THRESHOLD:g} + compensation"
# The scores a non-learned top-k is cut by, the label each is reported
# under and the colour of its bars: RadSplat's peak blending weight and
# the two sensitivities of Hanson et al. (anyprune.gaussians.sensitivity),
# whose papers prune an optimized scene and fine-tune it, which is what
# the post-optimization rows do to a predicted field
SCORES = {
    "radsplat": ("RadSplat", SERIES_COLORS[3]),
    "speedy-splat": ("Speedy-Splat", SERIES_COLORS[0]),
    "pup": ("PUP 3D-GS", SERIES_COLORS[6]),
    "mini-splatting": ("Mini-Splatting", SERIES_COLORS[4]),
    "lightgaussian": ("LightGaussian", SERIES_COLORS[7]),
    "refine": ("REFINE", SERIES_COLORS[5]),
    "gaussianpop": ("GaussianPOP", SERIES_COLORS[1]),
}
# Where each score's own pipeline cuts a field, for --own-rules: a
# threshold on the score or the share of the field it keeps, one shot.
# RadSplat drops every weight under 0.01 (its t_prune). Speedy-Splat
# prunes 80% three times during densification and 30% five times after
# it, ending ~10x under 3D-GS (its paper's 10.6x), and PUP 3D-GS 80%
# then 50% (scripts/full_pruning_pipeline.sh), i.e. 90%: a predicted
# field is pruned once, so both keep 10%. Mini-Splatting's first
# simplification draws, without replacement and with probability
# proportional to its score, sampling_factor times the Gaussians that
# score above zero (ms/train.py), 0.5 on Mip-NeRF 360. LightGaussian
# prunes the lowest 66% of its global significance once
# (scripts/run_prune_finetune.sh, --prune_percent 0.66). REFINE prunes
# once, 10-70% in its paper, 50% in its README. GaussianPOP cuts 65-70%
# (75% outdoors) after training, in GAUSSIANPOP_CYCLES steps.
OWN_RULES = {
    "radsplat": ("threshold", THRESHOLD),
    "speedy-splat": ("keep", 0.10),
    "pup": ("keep", 0.10),
    "mini-splatting": ("sample", 0.5),
    "lightgaussian": ("keep", 0.34),
    "refine": ("keep", 0.5),
    "gaussianpop": ("cycles", 0.3),
}
SUFFIX_REFINED = " + SplatFormer"
SUFFIX_OPTIMIZED = " + post-optimization"
MASK_COLORS = [SERIES_COLORS[2], SERIES_COLORS[5], SERIES_COLORS[4], SERIES_COLORS[1]]


@dataclass
class LearnedMask:
    """A learned mask refiner under test, with the score its mask head reads."""
    label: str
    checkpoint: Path
    score: str
    reference: float
    model: Optional[SplatFormer] = None

    @property
    def method(self) -> str:
        return f"learned mask ({self.label}){SUFFIX_REFINED}"


def method_topk(score: str, suffix: str = "", at: Optional[str] = None, compensate: bool = True) -> str:
    """The row of a top-k by 'score', compensated or not, with what came after and whose count it was cut at."""
    return (
        f"{SCORES[score][0]} top-k" + (" + compensation" if compensate else "") + suffix
        + ("" if at is None else f" [at {at}'s count]")
    )


def method_own_rule(score: str, compensate: bool) -> str:
    """The row of a score cut where its own paper cuts it."""
    kind, value = OWN_RULES[score]
    rule = {
        "threshold": f"t={value:g}", "keep": f"keep {value:.0%}", "sample": f"draw {value:g}x",
        "cycles": f"keep {value:.0%} in {GAUSSIANPOP_CYCLES} cycles",
    }[kind]
    return f"{SCORES[score][0]} own rule ({rule})" + (" + compensation" if compensate else "")


def method_cycles(compensate: bool = True) -> str:
    """The row of GaussianPOP's cycled cut at the primary mask's count."""
    return f"{SCORES['gaussianpop'][0]} {GAUSSIANPOP_CYCLES} cycles" + (" + compensation" if compensate else "")


def cycled_indices(field: Gaussians, context: ViewSet, count: int, generator: Generator) -> Tensor:
    """What GaussianPOP's cycled cut keeps of a field at 'count', as indices into it."""
    return gaussianpop_prune(
        field, count, context.poses, context.intrinsics, context.image_shape, generator=generator,
        batches_per_pass=PRUNING_BATCHES_PER_PASS, max_intersections=DEVICE_MAX_INTERSECTIONS,
    )


def top_indices(values: Tensor, count: int, generator: Generator) -> Tensor:
    """
    The indices of the 'count' largest values, ties broken at random
    (Pruner.order): a score that hands out many zeros, Mini-Splatting's
    above all, would otherwise fill a count past its non-zero set in the
    order the field was predicted in, i.e. from one corner of the images.
    """
    shuffled = torch.randperm(values.numel(), generator=generator).to(values.device)
    return shuffled[torch.topk(values[shuffled], count, sorted=False).indices]


def own_rule_indices(score: str, values: Tensor, generator: Generator) -> Tensor:
    """What a score's own rule keeps of a field, as indices into it."""
    kind, value = OWN_RULES[score]
    if kind == "threshold":
        kept = torch.nonzero(values >= value).squeeze(1)
        return kept if kept.numel() > 0 else values.argmax().reshape(1)
    if kind == "sample":
        # A draw without replacement in proportion to the score, as the
        # top of the log score perturbed by a Gumbel (Pruner.order)
        count = max(1, int(value * (values > 0).sum().item()))
        noise = torch.rand(values.numel(), generator=generator).to(values.device).clamp_min(torch.finfo(values.dtype).tiny)
        keys = values.float().log() - (-noise.log()).log()
        return torch.topk(keys, count, sorted=False).indices
    return top_indices(values, max(1, round(value * values.numel())), generator)


def style_of(method: str, masks: Sequence[LearnedMask]):
    """(label, colour, hatch) of a method's bars: a hue per rule, hatched when refined or optimized."""
    hatch = "xx" if SUFFIX_OPTIMIZED in method else ("//" if SUFFIX_REFINED in method else None)
    if method == METHOD_FULL:
        return method, REFERENCE_COLOR, None
    if method == METHOD_THRESHOLD_COMP:
        return method, SERIES_COLORS[1], None
    for i, mask in enumerate(masks):
        if method.startswith(f"learned mask ({mask.label})"):
            return method, MASK_COLORS[i % len(MASK_COLORS)], hatch
    for score, (label, color) in SCORES.items():
        if method.startswith(label):
            return method, color, hatch
    return method, SERIES_COLORS[7 % len(SERIES_COLORS)], hatch


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


def build_refiner(
    checkpoint: Path, device: torch.device, mask_head: bool, importance: bool,
    reference: float = IMPORTANCE_REFERENCE,
) -> SplatFormer:
    return SplatFormer(
        str(checkpoint), quiet=True, batch_statistics=REFINER_BATCH_STATISTICS,
        importance_input=importance, mask_head=mask_head,
        importance_reference=reference, importance_eps=IMPORTANCE_EPS,
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


def measure_scores(field: Gaussians, context: ViewSet, which: Sequence[str]) -> Dict[str, Tensor]:
    """
    The scores named in 'which' (keys of SCORES), each one number per
    Gaussian, measured over the context views: one sweep for RadSplat's,
    Mini-Splatting's and LightGaussian's, one more for the sensitivities
    and GaussianPOP's error, the Fisher only when PUP's is asked for, and
    none for REFINE's, which only reads where the cameras are.
    """
    scores = {}
    kwargs = dict(batches_per_pass=PRUNING_BATCHES_PER_PASS, max_intersections=DEVICE_MAX_INTERSECTIONS)
    if "radsplat" in which or "mini-splatting" in which or "lightgaussian" in which:
        measured = blending_weights(field, context.poses, context.intrinsics, context.image_shape, **kwargs)
        scores["radsplat"] = radsplat_score(measured)
        if "mini-splatting" in which:
            scores["mini-splatting"] = mini_splatting_score(measured)
        if "lightgaussian" in which:
            scores["lightgaussian"] = lightgaussian_score(measured, field)
        del measured
    if "speedy-splat" in which or "pup" in which or "gaussianpop" in which:
        sensitivities = sensitivity_scores(
            field, context.poses, context.intrinsics, context.image_shape, fisher="pup" in which, **kwargs,
        )
        if "speedy-splat" in which:
            scores["speedy-splat"] = speedy_splat_score(sensitivities)
        if "gaussianpop" in which:
            scores["gaussianpop"] = gaussianpop_score(sensitivities)
        if "pup" in which:
            scores["pup"] = pup_score(sensitivities)
        del sensitivities
    if "refine" in which:
        scores["refine"] = refine_score(field, context.poses)
    return scores


def evaluate_scene(
    masks: Sequence[LearnedMask], baseline_scores: Sequence[str], topk_refiners: Dict[str, SplatFormer],
    post_steps: int, scorer: Scorer, reconstructor, scene, context_idx, test_idx, scene_idx: int,
    full_post_optimization: bool = False, raw_topk: bool = False, own_rules: bool = False,
) -> List[dict]:
    """
    Every method on one scene at one view count. The first mask is the
    primary one: every baseline is cut at the count it kept (as refined,
    without compensation), raw, through the refiner trained on such
    fields when there is one for its score, and after post_steps of
    per-scene optimization when asked for; every other mask is scored
    at its own count against the raw top-k of its own score there. The
    threshold keeps whatever scores above it. With 'raw_topk' every
    baseline is also scored at the primary count without compensation,
    and with 'own_rules' at its own rule's count (OWN_RULES), with and
    without, in place of the threshold row.
    """
    with torch.no_grad():
        reconstruction = reconstruct(reconstructor, scene, context_idx, test_idx)
        field, context = reconstruction.gaussians, reconstruction.context
        predicted = field.num_gaussians
        scores = measure_scores(
            field, context, set(baseline_scores) | {mask.score for mask in masks} | {"radsplat"},
        )
    above = int((scores["radsplat"] >= THRESHOLD).sum().item())

    def optimized(name: str, gaussians: Gaussians) -> dict:
        tuned = fine_tune(
            gaussians, context.poses, context.intrinsics, context.images, post_steps,
            generator=Generator().manual_seed(SEED + scene_idx),
        )
        record = {"method": name + SUFFIX_OPTIMIZED, "num_gaussians": tuned.num_gaussians, **scorer(tuned, reconstruction)}
        del tuned
        return record

    records = [{"method": METHOD_FULL, "num_gaussians": predicted, **scorer(field, reconstruction)}]
    if post_steps and full_post_optimization:
        records.append(optimized(METHOD_FULL, field))

    # What the card holds of the prediction, cut by the score each mask
    # reads, refined and pruned by it
    counts = {}
    for mask in masks:
        score = scores[mask.score]
        handed, handed_score = field, score
        if predicted > DEVICE_MAX_GAUSSIANS:
            top = torch.topk(score, DEVICE_MAX_GAUSSIANS, sorted=False).indices
            handed, handed_score = field[top], score[top]
        with torch.no_grad():
            pruned = refine_and_prune(mask.model, build_rule(False), handed, handed_score)
        survivors = pruned.kept
        counts[mask.label] = survivors.num_gaussians
        records.append({
            "method": mask.method, "num_gaussians": survivors.num_gaussians,
            "input_gaussians": handed.num_gaussians,
            "soft_keep": pruned.soft["mask"].mean().item(),
            "scale_ratio": (pruned.refined.scales.float().mean() / handed.scales.float().mean()).item(),
            **scorer(survivors, reconstruction),
        })
        if post_steps:
            records.append(optimized(mask.method, survivors))
        del pruned, survivors, handed, handed_score

    # The threshold, at its own count, or every baseline at its own rule's
    with torch.no_grad():
        if own_rules:
            for score in baseline_scores:
                generator = Generator().manual_seed(SEED + scene_idx)
                if OWN_RULES[score][0] == "cycles":
                    kept = field[cycled_indices(field, context, max(1, round(OWN_RULES[score][1] * predicted)), generator)]
                else:
                    kept = field[own_rule_indices(score, scores[score], generator)]
                for compensate in (False, True):
                    cut = compensated(kept, predicted) if compensate else kept
                    records.append({
                        "method": method_own_rule(score, compensate), "num_gaussians": cut.num_gaussians,
                        **scorer(cut, reconstruction),
                    })
                    del cut
                del kept
        else:
            kept = torch.nonzero(scores["radsplat"] >= THRESHOLD).squeeze(1)
            if kept.numel() == 0:
                kept = scores["radsplat"].argmax().reshape(1)
            thresholded = compensated(field[kept], predicted)
            records.append({
                "method": METHOD_THRESHOLD_COMP, "num_gaussians": thresholded.num_gaussians,
                **scorer(thresholded, reconstruction),
            })
            del thresholded

    # The top of each score at the primary mask's count, compensated,
    # raw, refined and optimized; then the raw top of each other mask's
    # own score at its count
    with torch.no_grad():
        for i, mask in enumerate(masks):
            count = counts[mask.label]
            for score in (baseline_scores if i == 0 else [mask.score]):
                at = None if i == 0 else mask.label
                top = field[top_indices(scores[score], count, Generator().manual_seed(SEED + scene_idx))]
                if raw_topk:
                    records.append({
                        "method": method_topk(score, at=at, compensate=False), "num_gaussians": top.num_gaussians,
                        **scorer(top, reconstruction),
                    })
                matched = compensated(top, predicted)
                del top
                records.append({
                    "method": method_topk(score, at=at), "num_gaussians": matched.num_gaussians,
                    **scorer(matched, reconstruction),
                })
                if i > 0:
                    del matched
                    continue
                if score == "gaussianpop":
                    cycled = field[cycled_indices(field, context, count, Generator().manual_seed(SEED + scene_idx))]
                    if raw_topk:
                        records.append({
                            "method": method_cycles(compensate=False), "num_gaussians": cycled.num_gaussians,
                            **scorer(cycled, reconstruction),
                        })
                    cycled = compensated(cycled, predicted)
                    records.append({
                        "method": method_cycles(), "num_gaussians": cycled.num_gaussians,
                        **scorer(cycled, reconstruction),
                    })
                    del cycled
                refiner = topk_refiners.get(score)
                if refiner is not None:
                    try:
                        with torch.cuda.amp.autocast(enabled=False):
                            refined = refiner(matched)
                        records.append({
                            "method": method_topk(score, SUFFIX_REFINED), "num_gaussians": refined.num_gaussians,
                            **scorer(refined, reconstruction),
                        })
                        del refined
                    except RuntimeError as error:
                        if not out_of_memory(error):
                            raise
                        error.__traceback__ = None
                        torch.cuda.empty_cache()
                        tqdm.write(f"  {count:,} Gaussians did not fit {method_topk(score, SUFFIX_REFINED)}, skipped")
                if post_steps:
                    with torch.enable_grad():
                        records.append(optimized(method_topk(score), matched))
                del matched
    del field, scores
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
    for method, r in cells.items():
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


def plot_counts(records: Sequence[dict], context_views: Sequence[int], title: str, mask_method: str) -> Figure:
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
        cells = [r for r in records if r["context_views"] == views and r["method"] == mask_method]
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


def parse_mask(spec: str) -> LearnedMask:
    """label=path[,score[,reference]] on the command line."""
    label, _, rest = spec.partition("=")
    parts = rest.split(",")
    assert label and parts[0], f"A learned mask is given as label=path[,score[,reference]], got {spec!r}"
    score = parts[1] if len(parts) > 1 else "radsplat"
    assert score in SCORES, f"The score of {spec!r} has to be one of {list(SCORES)}"
    reference = float(parts[2]) if len(parts) > 2 else (
        SPEEDY_IMPORTANCE_REFERENCE if score == "speedy-splat" else IMPORTANCE_REFERENCE
    )
    return LearnedMask(label, Path(parts[0]), score, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks", nargs="+", default=[f"Speedy prior={LEARNED_CHECKPOINT},speedy-splat"],
                        metavar="LABEL=PATH[,SCORE[,REFERENCE]]",
                        help="the learned mask refiners, each with the score its mask head reads "
                             f"(one of {list(SCORES)}, radsplat by default) and the reference the score is "
                             "centred on; the first is the primary one every baseline is count-matched to")
    parser.add_argument("--baseline-scores", nargs="+", default=["radsplat", "speedy-splat"], choices=list(SCORES),
                        help="the scores the non-learned top-k baselines are cut by")
    parser.add_argument("--topk-checkpoints", nargs="*", default=[f"radsplat={TOPK_COMPENSATED_CHECKPOINT}"],
                        metavar="SCORE=PATH", help="a refiner trained on compensated top-k fields, per score")
    parser.add_argument("--post-optimize", type=int, default=0, metavar="STEPS",
                        help="also score every mask and every baseline after this many steps of per-scene "
                             "3DGS optimization on the context views (anyprune.gaussians.optimization)")
    parser.add_argument("--full-post-optimize", action="store_true",
                        help="with --post-optimize, also the whole prediction after as many steps: what the "
                             "optimization is worth at the full count")
    parser.add_argument("--raw-topk", action="store_true",
                        help="also score every baseline top-k at the mask's count without compensation")
    parser.add_argument("--own-rules", action="store_true",
                        help="score every baseline at the count its own rule keeps (OWN_RULES), with and "
                             "without compensation, in place of the threshold row")
    parser.add_argument("--importance", action="store_true",
                        help="the learned refiners were trained with the importance input")
    parser.add_argument("--run-name", default=None, help="what to file the results under; the primary checkpoint's stem by default")
    parser.add_argument("--num-scenes", type=int, default=None, help="how many test scenes, all by default")
    parser.add_argument("--context-views", type=int, nargs="+", default=EVAL_CONTEXT_VIEWS)
    parser.add_argument("--wandb", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()

    masks = [parse_mask(spec) for spec in args.masks]
    for mask in masks:
        assert mask.checkpoint.exists(), f"No learned mask refiner at {mask.checkpoint}"
    assert len({mask.label for mask in masks}) == len(masks), "Every mask needs its own label"
    run_name = args.run_name or masks[0].checkpoint.stem

    load_dotenv()
    set_rng_seed(SEED, deterministic=False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    dataset = DL3DVDataset(str(DL3DV_ROOT_DIR), DL3DV_IMAGES_SUBDIR)
    test_scenes = split_scenes(len(dataset), SPLITS, SPLIT_SEED)["test"]
    if args.num_scenes is not None:
        test_scenes = test_scenes[:args.num_scenes]

    for mask in masks:
        mask.model = build_refiner(
            mask.checkpoint, device, mask_head=True, importance=args.importance, reference=mask.reference,
        )
    topk_refiners, checkpoints = {}, {mask.method: str(mask.checkpoint) for mask in masks}
    for spec in args.topk_checkpoints:
        score, _, path = spec.partition("=")
        assert score in SCORES and path, f"A top-k refiner is given as score=path, got {spec!r}"
        path = Path(path)
        if not path.exists():
            print(f"Skipping {method_topk(score, SUFFIX_REFINED)!r}: no checkpoint at {path}")
            continue
        topk_refiners[score] = build_refiner(path, device, mask_head=False, importance=False)
        checkpoints[method_topk(score, SUFFIX_REFINED)] = str(path)
    reconstructor = build_reconstructor(RECONSTRUCTOR, ANYSPLAT_CHECKPOINT, YONOSPLAT_CHECKPOINT).to(device)
    scorer = Scorer()

    output_dir = EVAL_OUTPUT_ROOT / run_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    window = 2 * max(args.context_views)
    print(
        f"Evaluating the learned masks {[mask.label for mask in masks]} on {len(test_scenes)} scenes of the "
        f"DL3DV test split, windows of {window} frames, from {args.context_views} context views, against the "
        f"{[SCORES[s][0] for s in args.baseline_scores]} top-k at the count of {masks[0].label!r}"
        + (f", each also after {args.post_optimize} steps of per-scene optimization" if args.post_optimize else "")
        + f". Writing to {output_dir}"
    )
    use_wandb = args.wandb != "disabled"
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT, mode=args.wandb, name=f"{run_name}-eval", job_type="eval",
            config={"checkpoints": checkpoints, "num_scenes": len(test_scenes),
                    "context_views": args.context_views, "importance_input": args.importance,
                    "baseline_scores": args.baseline_scores, "post_optimize": args.post_optimize},
            settings=wandb.Settings(x_disable_stats=True),
        )

    def write_results():
        (output_dir / "results.json").write_text(json.dumps({
            "checkpoints": checkpoints, "importance_input": args.importance,
            "masks": [{"label": m.label, "score": m.score, "reference": m.reference} for m in masks],
            "baseline_scores": args.baseline_scores, "post_optimize": args.post_optimize,
            "raw_topk": args.raw_topk,
            "own_rules": {s: OWN_RULES[s] for s in args.baseline_scores} if args.own_rules else None,
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
                    masks, args.baseline_scores, topk_refiners, args.post_optimize, scorer, reconstructor,
                    scene, context_idx[:views], test_idx[:views], scene_idx,
                    full_post_optimization=args.full_post_optimize,
                    raw_topk=args.raw_topk, own_rules=args.own_rules,
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
    report = [
        f"Learned masks {[mask.label for mask in masks]}, {len(test_scenes)} DL3DV test scenes; "
        f"the baselines at the count of {masks[0].label!r}"
    ]
    for views in args.context_views:
        report += [f"\n{views} context views\n", table(summary, views)]
    report = "\n".join(report)
    print(report)
    (output_dir / "summary.txt").write_text(report)

    methods = list(dict.fromkeys(r["method"] for r in summary))
    drawn = [style_of(method, masks) for method in methods]
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
        mask_method=masks[0].method,
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
