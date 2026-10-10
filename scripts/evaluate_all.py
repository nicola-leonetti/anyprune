"""
The whole pruning benchmark in one command: every pre-existing pruning score, the uniform draw and
AnyPrune, on every reconstructor and dataset of configs/evaluate_all.yaml, each cut to the same fixed
shares of the same predicted field.

    python scripts/evaluate_all.py                                   # everything
    python scripts/evaluate_all.py run.reconstructors=[YoNoSplat] run.views=[2,4]
    python scripts/evaluate_all.py datasets.dl3dv.root=/data/dl3dv datasets.zeroshot.root=/data/zeroshot
    python scripts/evaluate_all.py --dry-run                         # list what would run
    python scripts/plot_evaluate_all.py outputs/evaluate-all         # figures and tables

What the other machine needs next to the repository (with its submodules and environment):
    checkpoints/anyprune/final/*.pth   the four AnyPrune models (checked against SHA256SUMS)
    DL3DV                              the same dataset as for training (images_8)
    zeroshot-270.tar                   the zero-shot dataset, unpacked; set datasets.zeroshot.root
The reconstructors' own weights are fetched from Hugging Face on first use.

Protocol (the one of every benchmark so far): for each scene, a 128-frame window drawn with seed +
scene index, context and test frames alternating, the first V of each. The reconstructor predicts one
field from the V context views. Every score is measured once over all V context views and the field is
cut to the top 10..90% of it, raw, with ties broken at random and every share a prefix of the same order.
AnyPrune makes one refiner forward over the field, which gives the refined field and one logit per
Gaussian; per model it records
    <name>          the top of the refined field by logit (at 1.0 the whole refined field)
    <name>-logits   the top of the predicted field by the same logits, raw like the other scores
    <name>-own      the refined field at the model's own cut, logit > 0 (one record, at its own share)
    <name>-whole    with also_whole: the same with the refiner's whole residual
PSNR, SSIM and LPIPS are rendered on the context views ('self_*') and the held-out test views ('nvs_*'),
at most max_scored_views of each.

Output, under output_dir:
    cells/<dataset>/<reconstructor>/<V>v/scene-<index>.json
        one per (dataset, reconstructor, views, scene): the records of every method, the field's size,
        timings, peak memory, and what did not fit ('skipped'). A rerun skips every complete cell
        measured under the same settings, so the run can be stopped and resumed, and a cell that was
        incomplete (something ran out of memory) is tried again.
    results.json    the mean, std and SEM over scenes of every (dataset, reconstructor, views, method,
                    fraction, metric)
    skipped.json    every cell or method that did not fit, with the error
    config.yaml     the resolved configuration; provenance/ the git state, checkpoint hashes, versions
A run that skipped anything for lack of memory exits with code 3.
"""
import argparse
import hashlib
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
from omegaconf import OmegaConf
from torch import Generator, Tensor
from tqdm import tqdm

import eval_learned as el
from anyprune.datasets import DL3DVDataset, split_scenes
from anyprune.gaussians import Gaussians
from anyprune.models import build_reconstructor
from anyprune.models.utils import build_anysplat_covariance
from anyprune.training import reconstruct, sample_view_indices
from anyprune.training.reconstruction import Reconstruction
from anyprune.utils import load_dotenv, out_of_memory, set_rng_seed

CONFIG = ROOT / "configs" / "evaluate_all.yaml"
FULL = "full field"
METRICS = ("psnr", "ssim", "lpips")
BLOCKS = ("self", "nvs")
# The halo around each core of a chunked refinement, as a share of the core's 1-99% box
HALO = 0.1


# --- The cut ---

def order(values: Tensor, generator: Generator) -> Tensor:
    """Every index of a score, best first, ties in a random order: the top-k of every count is a prefix."""
    values = torch.nan_to_num(values.float(), nan=float("-inf"))
    shuffled = torch.randperm(values.numel(), generator=generator).to(values.device)
    return shuffled[torch.sort(values[shuffled], descending=True, stable=True).indices]


def scored(reconstruction: Reconstruction, max_views) -> Reconstruction:
    """The reconstruction the metrics are rendered on: at most 'max_views' of each half, evenly spaced."""
    def thin(views):
        if max_views is None or len(views) <= max_views:
            return views
        index = torch.linspace(0, len(views) - 1, max_views).round().long().to(views.images.device)
        return replace(views, images=views.images[index], poses=views.poses[index], intrinsics=views.intrinsics[index])

    return replace(reconstruction, context=thin(reconstruction.context), test=thin(reconstruction.test))


# --- The refiner, whole or in spatial chunks ---

def _params(g: Gaussians, d_sh: int = 4) -> dict:
    harmonics = torch.nn.functional.pad(g.harmonics, (0, d_sh - g.harmonics.shape[-1]))
    return {"means": g.means, "scales": g.scales.clamp_min(1e-12).log(),
            "opacities": g.opacities.clamp(1e-6, 1 - 1e-6).logit(), "rotations": g.rotations, "harmonics": harmonics}


def _from_params(p: dict) -> Gaussians:
    rotations = p["rotations"] / p["rotations"].norm(dim=-1, keepdim=True)
    scales = p["scales"].exp()
    return Gaussians(means=p["means"], covariances=build_anysplat_covariance(scales, rotations),
                     harmonics=p["harmonics"], opacities=torch.sigmoid(p["opacities"]), scales=scales, rotations=rotations)


def _median_cores(means: Tensor, index: Tensor, target: int) -> list:
    """Recursive median cuts along the longest axis of the 1-99% box until every core is <= target."""
    if index.numel() <= target:
        return [index]
    pts = means[index]
    lo, hi = torch.quantile(pts[:: max(1, pts.shape[0] // 100_000)], torch.tensor([0.01, 0.99], device=pts.device), dim=0)
    axis = int((hi - lo).argmax())
    ranked = torch.argsort(pts[:, axis])
    half = index.numel() // 2
    return _median_cores(means, index[ranked[:half]], target) + _median_cores(means, index[ranked[half:]], target)


@torch.no_grad()
def _refine_chunked(model, field: Gaussians, score: Tensor, target: int, features=None):
    """Refine each core with a halo around it, normalized on its own; the stitched field and logits."""
    n = field.num_gaussians
    out = {k: v.clone() for k, v in _params(field).items()}
    logits = torch.empty(n, device=field.device)
    cores = _median_cores(field.means, torch.arange(n, device=field.device), target)
    for core in cores:
        pts = field.means[core]
        lo, hi = torch.quantile(pts[:: max(1, pts.shape[0] // 100_000)], torch.tensor([0.01, 0.99], device=pts.device), dim=0)
        margin = HALO * (hi - lo)
        inside = ((field.means >= lo - margin) & (field.means <= hi + margin)).all(-1)
        inside[core] = True
        members = torch.nonzero(inside).squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            # The residual is scaled by the size of the whole field, not of the chunk
            refined, lg = model.refine(field[members], score[members], field_gaussians=n,
                                       image_features=None if features is None else features[members])
        position = torch.full((n,), -1, dtype=torch.long, device=field.device)
        position[members] = torch.arange(members.numel(), device=field.device)
        local = position[core]
        p = _params(refined)
        for k in out:
            out[k][core] = p[k][local]
        logits[core] = lg[local]
    return _from_params(out), logits, len(cores)


def refine_field(model, field: Gaussians, score: Tensor, chunk_gaussians, features=None):
    """
    The refined field, its logits and how it was refined ('whole' or 'chunked-<cores>'), or raise an OOM.
    'features' are an image-conditioned model's features lifted over the whole field (model.image_features),
    which a chunked refinement cuts along with the field.
    """
    refined = None
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            refined, logits = model.refine(field, score, image_features=features)
    except RuntimeError as error:
        if not out_of_memory(error) or chunk_gaussians is None:
            raise
        error.__traceback__ = None
    if refined is not None:
        return refined, logits, "whole"
    # Out of the except block, so that the failed forward's memory is released first
    torch.cuda.empty_cache()
    refined, logits, cores = _refine_chunked(model, field, score, chunk_gaussians, features)
    return refined, logits, f"chunked-{cores}"


# --- One cell ---

class Skips:
    """What did not fit, by cell and method, with the error; an incomplete cell is retried on a rerun."""

    def __init__(self):
        self.entries = []

    def add(self, where: dict, what: str, error: BaseException):
        message = str(error).split("\n")[0][:300]
        self.entries.append({**where, "what": what, "error": message})
        tqdm.write(f"  !! skipped {what} at {where}: {message}", file=sys.stderr)
        torch.cuda.empty_cache()


def guarded(skips, where: dict, what: str, function, *args, **kwargs):
    """function(*args, **kwargs), or None when it ran out of GPU memory (recorded in 'skips' unless None)."""
    try:
        return function(*args, **kwargs)
    except RuntimeError as error:
        if not out_of_memory(error):
            raise
        error.__traceback__ = None
    if skips is not None:
        skips.add(where, what, torch.cuda.OutOfMemoryError("out of memory"))
    torch.cuda.empty_cache()
    return None


def measure(field, context, methods, skips, where):
    """The scores of 'methods', all in one pass when they fit, one at a time when they do not."""
    wanted = [m for m in methods if m != "uniform"]
    if not wanted:
        return {}
    # All in one pass first, which shares the sweeps; quietly, since a failure is retried score by score
    scores = guarded(None, where, "scores", el.measure_scores, field, context, wanted)
    if scores is not None:
        return scores
    torch.cuda.empty_cache()
    scores = {}
    for method in wanted:
        one = guarded(skips, where, f"score {method}", el.measure_scores, field, context, [method])
        if one is not None:
            scores[method] = one[method]
    return scores


def _reconstruct(reconstructor, scene, context_idx, test_idx):
    with torch.no_grad():
        return reconstruct(reconstructor, scene, context_idx, test_idx)


def _lift(model, field, context):
    """A model's image features over the whole field; on running out of memory, again one view per rasterizer pass."""
    features = guarded(None, None, "image features", model.image_features, field, context)
    if features is not None:
        return features
    tqdm.write("  !! the image features did not fit, lifting them again one view at a time", file=sys.stderr)
    torch.cuda.empty_cache()
    model._extractor.to(field.device)
    with torch.cuda.amp.autocast(enabled=False):
        return model._extractor.lift(field, context.images, context.poses, context.intrinsics, views_per_pass=1)


def _move_models(models, device):
    for _, model in models:
        model.to(device)
        if model.image_conditioned:
            model._extractor.to(device)


def evaluate_cell(cfg, reconstructor, models, scorer, dataset, scene_idx, views, device, where, reuse=None) -> dict:
    """
    One cell. 'reuse' is a stored cell of the same scene and settings whose records are kept as they are
    (its unpruned field and baselines, and the models not in 'models'): only 'models' are run, and the
    unpruned field is rendered again to check that the field is the one the stored records were cut from.
    """
    protocol = cfg.protocol
    skips = Skips()
    # A scene shorter than the window (the strided RE10K / ACID clips) is read whole; it still needs 2 * views frames
    num_frames = dataset.num_frames(scene_idx)
    frames, context_idx, test_idx = sample_view_indices(
        num_frames, min(protocol.window, num_frames // 2 * 2),
        generator=Generator().manual_seed(protocol.seed + scene_idx),
    )
    assert len(context_idx) >= views, f"scene {scene_idx} has {num_frames} frames, too few for {views} views"
    scene = {name: value.to(device) for name, value in dataset.get_frames(scene_idx, frames).items()}
    timings = {}
    start = time.perf_counter()
    reconstructor.to(device)
    reconstruction = guarded(None, where, "reconstruction", _reconstruct, reconstructor, scene, context_idx[:views],
                             test_idx[:views])
    offloaded = reconstruction is None and bool(models)
    if offloaded:
        # Out of memory next to the refiners and the image extractor (VolSplat at 4 views): again with them off the card
        tqdm.write(f"  !! {where}: the reconstruction did not fit next to the refiners, retrying with them on the CPU",
                   file=sys.stderr)
        _move_models(models, "cpu")
        torch.cuda.empty_cache()
        reconstruction = _reconstruct(reconstructor, scene, context_idx[:views], test_idx[:views])
    elif reconstruction is None:
        reconstruction = _reconstruct(reconstructor, scene, context_idx[:views], test_idx[:views])
    del scene
    field, context = reconstruction.gaussians, reconstruction.context
    predicted = field.num_gaussians
    if offloaded:
        reconstructor.to("cpu")
        torch.cuda.empty_cache()
        _move_models(models, device)
    elif predicted > cfg.device.offload_reconstructor_above:
        reconstructor.to("cpu")
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    timings["reconstruct"] = time.perf_counter() - start
    scoring = scored(reconstruction, protocol.max_scored_views)
    fractions = list(protocol.fractions)

    def score_cut(method, source, ranked, fraction):
        kept = source if ranked is None else source[ranked[:max(1, round(fraction * predicted))]]
        result = guarded(skips, where, f"render {method} {fraction:.0%}", scorer, kept, scoring)
        if result is not None:
            records.append({"method": method, "fraction": fraction, "num_gaussians": kept.num_gaussians, **result})

    records = []
    start = time.perf_counter()
    score_cut(FULL, field, None, 1.0)
    reuse_check = None
    if reuse is not None:
        stored = next(r for r in reuse["records"] if r["method"] == FULL)
        fresh = next((r for r in records if r["method"] == FULL), None)
        reuse_check = {"from": reuse["path"], "methods": sorted({r["method"] for r in reuse["records"]})}
        if fresh is not None:
            reuse_check.update({f"{k}_diff": fresh[k] - stored[k] for k in ("self_psnr", "nvs_psnr")})
            if max(abs(reuse_check["self_psnr_diff"]), abs(reuse_check["nvs_psnr_diff"])) > 1e-3:
                tqdm.write(f"  !! {where}: the unpruned field renders {reuse_check['nvs_psnr_diff']:+.4f} dB NVS off "
                           f"the stored cell it reuses ({reuse['path']})", file=sys.stderr)
        records = [r for r in reuse["records"]]
    baselines = [] if reuse is not None else list(cfg.baselines)
    scores = measure(field, context, baselines + ["speedy-splat"] * bool(models), skips, where)
    for method in baselines:
        if method != "uniform" and method not in scores:
            continue
        generator = Generator().manual_seed(protocol.seed + scene_idx)
        ranked = (torch.randperm(predicted, generator=generator).to(device) if method == "uniform"
                  else order(scores[method], generator))
        for fraction in fractions:
            score_cut(method, field, ranked, fraction)
        del ranked
    torch.cuda.synchronize()
    timings["baselines"] = time.perf_counter() - start

    refinement = dict(reuse.get("refinement") or {}) if reuse is not None else {}
    lifted = {}
    for spec, model in models:
        name = spec.name
        if "speedy-splat" not in scores:
            skips.add(where, name, RuntimeError("its input score (Speedy-Splat) did not fit"))
            continue
        start = time.perf_counter()
        features = None
        if model.image_conditioned:
            # Lifted once over the whole field per set of sources, and shared by the models that read the same
            feature_sources = model._extractor.sources
            if feature_sources not in lifted:
                torch.cuda.empty_cache()
                lifted[feature_sources] = guarded(skips, where, f"{name} image features", _lift, model, field, context)
            features = lifted[feature_sources]
            if features is None:
                continue
        refined = guarded(skips, where, f"{name} refiner", refine_field, model, field, scores["speedy-splat"],
                          cfg.device.refine_chunk_gaussians, features)
        if refined is None:
            continue
        refined, logits, mode = refined
        whole = None
        if spec.also_whole and model.residual_scale_for(predicted) != 1.0:
            rule, model.residual_reference_gaussians = model.residual_reference_gaussians, None
            try:
                whole = guarded(skips, where, f"{name}-whole refiner", refine_field, model, field,
                                scores["speedy-splat"], cfg.device.refine_chunk_gaussians, features)
            finally:
                model.residual_reference_gaussians = rule
            whole = None if whole is None else whole[0]
        elif spec.also_whole:
            whole = refined
        own = logits > 0
        score_cut(name, refined, None, 1.0)
        kept = refined[own]
        result = guarded(skips, where, f"render {name}-own", scorer, kept, scoring)
        if result is not None:
            records.append({"method": f"{name}-own", "fraction": own.float().mean().item(),
                            "num_gaussians": kept.num_gaussians, **result})
        del kept
        ranked = order(logits, Generator().manual_seed(protocol.seed + scene_idx))
        sources = [(name, refined), (f"{name}-logits", field)]
        if whole is not None:
            score_cut(f"{name}-whole", whole, None, 1.0)
            sources.append((f"{name}-whole", whole))
        for method, source in sources:
            for fraction in fractions:
                score_cut(method, source, ranked, fraction)
        refinement[name] = {"mode": mode, "residual_scale": model.residual_scale_for(predicted),
                            "own_fraction": own.float().mean().item(), "seconds": time.perf_counter() - start}
        del refined, logits, whole, ranked, own, features
    torch.cuda.synchronize()
    del field, scores, reconstruction, scoring, lifted
    return {
        "reuse_check": reuse_check,
        "scene": scene_idx, "scene_name": Path(dataset.scenes[scene_idx]).name, "context_views": views,
        "frames": frames.tolist(), "predicted_gaussians": predicted, "refinement": refinement,
        "complete": not skips.entries, "skipped": skips.entries,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30, "timings": timings, "records": records,
    }


# --- Results ---

def aggregate(cells) -> list:
    """The mean, std and SEM over scenes of every (dataset, reconstructor, views, method, fraction)."""
    groups = defaultdict(list)
    for cell in cells:
        for r in cell["records"]:
            own = r["method"].endswith("-own")
            groups[(cell["dataset"], cell["reconstructor"], cell["context_views"], r["method"],
                    None if own else r["fraction"])].append(r)
    summary = []
    for (dataset, reconstructor, views, method, fraction), records in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4] or 0)):
        entry = {"dataset": dataset, "reconstructor": reconstructor, "context_views": views, "method": method,
                 "fraction": fraction if fraction is not None else sum(r["fraction"] for r in records) / len(records),
                 "own_cut": fraction is None, "num_scenes": len(records),
                 "num_gaussians": sum(r["num_gaussians"] for r in records) / len(records)}
        for block in BLOCKS:
            for metric in METRICS:
                values = [r[f"{block}_{metric}"] for r in records]
                mean = sum(values) / len(values)
                std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
                entry[f"{block}_{metric}"] = mean
                entry[f"{block}_{metric}_std"] = std
                entry[f"{block}_{metric}_sem"] = std / math.sqrt(len(values))
        summary.append(entry)
    return summary


def read_cells(output_dir: Path) -> list:
    cells = []
    for path in sorted((output_dir / "cells").glob("*/*/*v/scene-*.json")):
        cell = json.loads(path.read_text())
        cells.append(cell)
    return cells


# --- Setup ---

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(cfg, output_dir: Path, argv):
    target = output_dir / "provenance" / time.strftime("%Y%m%d-%H%M%S")
    target.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    (target / "command.txt").write_text(f"{' '.join(argv)}\nhost {platform.node()}\ngpu {gpu}\n"
                                        f"python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda}\n")
    (target / "git.txt").write_text(f"HEAD {git('rev-parse', 'HEAD').strip()}\n\n{git('status', '--short')}")
    (target / "repo.diff").write_text(git("diff", "HEAD"))
    hashes = []
    for reconstructor in cfg.run.reconstructors:
        for spec in model_specs(cfg, reconstructor):
            path = resolve(spec.checkpoint)
            hashes.append(f"{sha256(path) if path.is_file() else 'MISSING'}  {path}")
    (target / "checkpoints.txt").write_text("\n".join(hashes) + "\n")


def resolve(path) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else ROOT / path


def load_config(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="list the cells that would run, and stop")
    parser.add_argument("overrides", nargs="*", help="key=value overrides of the config (OmegaConf dot-list)")
    args = parser.parse_args(argv)
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    return cfg, args


def dataset_root(spec, views: int):
    """The dataset's root at a view count: 'root_by_views' (a copy at another frame stride) or 'root'."""
    by_views = {int(k): r for k, r in (spec.get("root_by_views") or {}).items()}
    return by_views.get(views, spec.root)


def more_roots(spec, views: int) -> list:
    """
    Further roots of the same dataset at a view count ('more_roots_by_views', or 'more_roots'), whose scenes
    are indexed after the root's: scenes added later in a root of their own leave every stored cell's index
    (and so its window) as it was.
    """
    by_views = {int(k): r for k, r in (spec.get("more_roots_by_views") or {}).items()}
    return [str(r) for r in by_views.get(views, spec.get("more_roots") or [])]


def scene_root(dataset, scene_idx: int) -> str:
    """The root directory the scene of an index was read from."""
    return str(Path(dataset.scenes[scene_idx]).parent)


def load_dataset(spec, protocol, root=None, more=()):
    """The dataset and the scene indices to evaluate: the repo's DL3DV split, or every scene."""
    dataset = DL3DVDataset(str(resolve(spec.root if root is None else root)), spec.images_subdir)
    for extra in more:
        if resolve(extra).is_dir():
            dataset.scenes += DL3DVDataset(str(resolve(extra)), spec.images_subdir).scenes
    scenes = list(range(len(dataset)))
    if spec.split is not None:
        scenes = split_scenes(len(dataset), el.SPLITS, el.SPLIT_SEED)[spec.split]
    if spec.num_scenes is not None:
        scenes = scenes[:spec.num_scenes]
    return dataset, scenes


def cell_settings(cfg, dataset_name, reconstructor_name, checkpoint, root) -> dict:
    """
    Everything the unpruned field and the baselines of a cell depend on: their stored records are reused only
    when this matches. Each AnyPrune model's records depend on model_settings as well.
    """
    dataset = cfg.datasets[dataset_name]
    return OmegaConf.to_container(OmegaConf.create({
        "protocol": cfg.protocol, "baselines": cfg.baselines,
        "dataset": {"root": str(resolve(root)), "images_subdir": dataset.images_subdir},
        "reconstructor": reconstructor_name, "reconstructor_checkpoint": checkpoint,
        "batch_statistics": el.REFINER_BATCH_STATISTICS, "deterministic_refiner": el.REFINER_DETERMINISTIC,
        "mask_reference": el.SPEEDY_IMPORTANCE_REFERENCE,
    }), resolve=True)


def model_specs(cfg, reconstructor_name) -> list:
    """The AnyPrune models of a reconstructor that this run evaluates (run.models, when given, names them)."""
    wanted = cfg.run.get("models")
    return [s for s in cfg.reconstructors[reconstructor_name].anyprune if wanted is None or s.name in wanted]


def model_settings(cfg, reconstructor_name) -> dict:
    """What each AnyPrune model's records depend on beyond the cell's settings, by name."""
    specs = [OmegaConf.to_container(s, resolve=True) for s in model_specs(cfg, reconstructor_name)]
    return {s["name"]: {"image_features": None, **s, "checkpoint": str(resolve(s["checkpoint"]))} for s in specs}


def stored_models(stored: dict) -> dict:
    """The model settings of a stored cell, including the cells written before model_settings existed."""
    if "model_settings" in stored:
        return stored["model_settings"]
    legacy = stored.get("settings", {}).get("anyprune") or []
    return {s["name"]: {"image_features": None, **s} for s in legacy}


def model_methods(name: str) -> set:
    return {name, f"{name}-own", f"{name}-logits", f"{name}-whole"}


def reusable(stored: dict, settings: dict) -> bool:
    """Whether a stored cell's field and baselines are those of 'settings' (cells imported from the earlier
    eval_fractions benchmarks record where they came from instead, and evaluate_all reproduces them bit for bit)."""
    # An incomplete cell is still reused when its unpruned field was rendered: what did not fit is
    # recorded in it, and on this card it would not fit again
    if not any(r["method"] == FULL for r in stored.get("records", [])):
        return False
    base = {k: v for k, v in stored.get("settings", {}).items() if k != "anyprune"}
    return base == settings or "imported_from" in base


def reuse_plan(cfg, settings, models, path: Path, sources: list):
    """
    (records to keep, model names to run) for one cell: everything from the cell already in the output
    directory when its settings match, else the unpruned field and baselines of the same cell in one of
    'sources' (cfg.reuse_from), else nothing (None, every model).
    """
    if path.exists():
        stored = json.loads(path.read_text())
        if reusable(stored, settings):
            done = stored_models(stored)
            todo = [n for n, s in models.items() if done.get(n) != s]
            stale = set().union(*(model_methods(n) for n in todo)) if todo else set()
            keep = {"path": str(path), "records": [r for r in stored["records"] if r["method"] not in stale],
                    "refinement": {k: v for k, v in (stored.get("refinement") or {}).items() if k not in todo},
                    "model_settings": {k: v for k, v in done.items() if k not in todo},
                    "baselines_from": stored.get("baselines_from", str(path))}
            return keep, todo
    for source in sources:
        candidate = resolve(source) / "cells" / path.relative_to(path.parents[3])
        if candidate.exists():
            stored = json.loads(candidate.read_text())
            if reusable(stored, settings):
                wanted = set(cfg.baselines) | {FULL}
                records = [r for r in stored["records"] if r["method"] in wanted]
                if {r["method"] for r in records} >= wanted - {"pup"} and any(r["method"] == FULL for r in records):
                    return {"path": str(candidate), "records": records, "refinement": {}, "model_settings": {},
                            "baselines_from": str(candidate)}, list(models)
    return None, list(models)


def plan(cfg):
    """(dataset, reconstructor, checkpoint, views) in the order they run."""
    jobs = []
    for reconstructor in cfg.run.reconstructors:
        spec = cfg.reconstructors[reconstructor]
        views = [v for v in spec.views if cfg.run.views is None or v in cfg.run.views]
        # Keys given on the command line arrive as strings
        by_views = {int(k): c for k, c in (spec.get("checkpoint_by_views") or {}).items()}
        for dataset in cfg.run.datasets:
            if dataset not in spec.datasets:
                continue
            for v in views:
                jobs.append((dataset, reconstructor, by_views.get(v, spec.checkpoint), v))
    return jobs


def build(reconstructor: str, checkpoint: str):
    kwargs = {"ReSplat": "resplat_checkpoint", "VolSplat": "volsplat_checkpoint"}
    if reconstructor in kwargs:
        return build_reconstructor(reconstructor, el.ANYSPLAT_CHECKPOINT, el.YONOSPLAT_CHECKPOINT,
                                   **{kwargs[reconstructor]: checkpoint})
    if reconstructor == "YoNoSplat":
        return build_reconstructor(reconstructor, el.ANYSPLAT_CHECKPOINT, checkpoint)
    return build_reconstructor(reconstructor, checkpoint, el.YONOSPLAT_CHECKPOINT)


def main(argv) -> int:
    cfg, args = load_config(argv)
    output_dir = resolve(cfg.output_dir)
    jobs = plan(cfg)
    datasets = {}
    key = lambda name, views: (name, str(dataset_root(cfg.datasets[name], views)), tuple(more_roots(cfg.datasets[name], views)))
    for name, _, _, views in jobs:
        if key(name, views) not in datasets:
            datasets[key(name, views)] = load_dataset(cfg.datasets[name], cfg.protocol, key(name, views)[1],
                                                      key(name, views)[2])
    reuse_sources = list(cfg.get("reuse_from") or [])

    def todo_of(dataset_name, reconstructor_name, checkpoint, views):
        """[(scene, reuse, model names to run)] of a job, the complete cells left out."""
        dataset, scenes = datasets[key(dataset_name, views)]
        models = model_settings(cfg, reconstructor_name)
        directory = output_dir / "cells" / dataset_name / reconstructor_name / f"{views}v"
        out = []
        for scene_idx in scenes:
            settings = cell_settings(cfg, dataset_name, reconstructor_name, checkpoint, scene_root(dataset, scene_idx))
            reuse, names = reuse_plan(cfg, settings, models, directory / f"scene-{scene_idx:04d}.json", reuse_sources)
            if reuse is None or names:
                out.append((scene_idx, reuse, names))
        return out

    if args.dry_run:
        for dataset, reconstructor, checkpoint, views in jobs:
            todo = todo_of(dataset, reconstructor, checkpoint, views)
            print(f"{dataset:9s} {reconstructor:19s} {views:3d} views x {len(datasets[key(dataset, views)][1])} scenes: "
                  f"{sum(r is None for _, r, _ in todo)} to run whole, {sum(r is not None for _, r, _ in todo)} "
                  f"to add models to  ({checkpoint})")
        return 0
    for reconstructor in cfg.run.reconstructors:
        for spec in model_specs(cfg, reconstructor):
            assert resolve(spec.checkpoint).is_file(), f"Missing AnyPrune checkpoint {resolve(spec.checkpoint)}"

    # The device and protocol settings eval_learned's helpers read at call time
    el.SEED = cfg.protocol.seed
    el.DEVICE_MAX_VIEWS_PER_RENDER = cfg.device.max_views_per_render
    el.DEVICE_MAX_INTERSECTIONS = int(cfg.device.max_intersections)
    el.PRUNING_BATCHES_PER_PASS = cfg.device.pruning_batches_per_pass
    load_dotenv()
    set_rng_seed(cfg.protocol.seed, deterministic=cfg.protocol.deterministic)
    if cfg.protocol.deterministic:
        # PUP's Fisher needs index_add_'s deterministic kernel to give the same order twice
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    write_provenance(cfg, output_dir, [sys.executable, *sys.argv])
    scorer = el.Scorer()

    skipped, current = [], (None, None, None)
    reconstructor_module, models = None, []
    for dataset_name, reconstructor_name, checkpoint, views in jobs:
        dataset, scenes = datasets[key(dataset_name, views)]
        directory = output_dir / "cells" / dataset_name / reconstructor_name / f"{views}v"
        specs = model_settings(cfg, reconstructor_name)
        todo = todo_of(dataset_name, reconstructor_name, checkpoint, views)
        if not todo:
            continue
        if current[:2] != (reconstructor_name, checkpoint):
            del reconstructor_module, models
            torch.cuda.empty_cache()
            reconstructor_module = build(reconstructor_name, checkpoint)
            models, extractors = [], {}
            for spec in model_specs(cfg, reconstructor_name):
                model = el.build_refiner(
                    resolve(spec.checkpoint), device, mask_head=True, importance=False,
                    reference=el.SPEEDY_IMPORTANCE_REFERENCE,
                    residual_rule=None if spec.residual_rule is None else tuple(spec.residual_rule),
                    size_reference=spec.size_reference, image_features=spec.get("image_features"),
                )
                if model.image_conditioned:
                    # One frozen extractor (DINOv2, ResNet-18) per set of sources, shared by the models
                    feature_sources = model._extractor.sources
                    object.__setattr__(model, "_extractor", extractors.setdefault(feature_sources, model._extractor))
                models.append((spec, model))
            current = (reconstructor_name, checkpoint, None)
        for scene_idx, reuse, names in tqdm(todo, desc=f"{dataset_name} {reconstructor_name} {views}v", unit="scene"):
            where = {"dataset": dataset_name, "reconstructor": reconstructor_name, "views": views, "scene": scene_idx}
            settings = cell_settings(cfg, dataset_name, reconstructor_name, checkpoint, scene_root(dataset, scene_idx))
            running = [(spec, model) for spec, model in models if spec.name in names]
            torch.cuda.reset_peak_memory_stats()
            try:
                cell = evaluate_cell(cfg, reconstructor_module, running, scorer, dataset, scene_idx, views, device,
                                     where, reuse)
            except RuntimeError as error:
                if not out_of_memory(error):
                    raise
                error.__traceback__ = None
                torch.cuda.empty_cache()
                skipped.append({**where, "what": "reconstruction", "error": "out of memory"})
                tqdm.write(f"  !! {where}: the reconstruction did not fit, cell skipped", file=sys.stderr)
                continue
            torch.cuda.empty_cache()
            skipped += cell["skipped"]
            directory.mkdir(parents=True, exist_ok=True)
            done = dict(reuse["model_settings"]) if reuse is not None else {}
            done.update({name: specs[name] for name in names if name in cell["refinement"]})
            (directory / f"scene-{scene_idx:04d}.json").write_text(json.dumps(
                {"settings": settings, "model_settings": done, "dataset": dataset_name,
                 "reconstructor": reconstructor_name, "reconstructor_checkpoint": checkpoint,
                 "baselines_from": None if reuse is None else reuse["baselines_from"], **cell}, indent=1))
            t = cell["timings"]
            tqdm.write(f"  {dataset_name} {reconstructor_name} {views}v scene {scene_idx}: "
                       f"{cell['predicted_gaussians']:,} Gaussians, reconstruct {t['reconstruct']:.0f}s, "
                       f"baselines {t['baselines']:.0f}s, "
                       + ", ".join(f"{n} {r['mode']} {r['seconds']:.0f}s" for n, r in cell["refinement"].items())
                       + f", peak {cell['peak_memory_gib']:.1f} GiB")

    (output_dir / "results.json").write_text(json.dumps(aggregate(read_cells(output_dir)), indent=1))
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=1))
    if skipped:
        print(f"!! {len(skipped)} cells or methods did not fit in GPU memory, see {output_dir / 'skipped.json'}",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
