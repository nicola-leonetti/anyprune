"""
Score refined Gaussians over the evaluation protocol of the report.

Every scene of the held-out split is fit once per context view count,
thinned down to each Gaussian budget in turn, refined, and rendered back
onto both the views the reconstructor saw and the ones held out from it,
where it is scored on PSNR, SSIM and LPIPS. A run produces the two
things the report asks for:

    - the curves of the three metrics against the context view count,
      one line per budget, for the variant being reported
    - the ablation table, every variant scored at the single cell
      ablation.gaussian_budget / ablation.context_views

A variant is a SplatFormer checkpoint: the run being reported, the runs
that dropped one piece of it, and SplatFormer as it was released, which
is the zero-shot row. Every variant sees the same scenes, the same views
and the same thinning, all drawn off generators seeded by the scene
alone, so that what separates two rows is the weights and nothing else.

Everything measured is logged to wandb as it is measured, written to
results.json, and printed as tables at the end.
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
import wandb
from omegaconf import OmegaConf
from tabulate import tabulate
from torch import Generator, Tensor
from tqdm import tqdm

from anyprune.datasets import DL3DVDataset, split_scenes
from anyprune.evaluation import LPIPS, psnr, ssim
from anyprune.models import RECONSTRUCTORS, SplatFormer, build_reconstructor
from anyprune.training import plan_context_views, reconstruct
from anyprune.utils import load_dotenv, out_of_memory, set_rng_seed
from anyprune.viz import plot_eval_curves


# The two halves of a scene a cell is scored on, the two stages of the
# pruning it is scored at, and the metrics it is scored with. Together
# they name every number a cell holds, as '<block>/<metric>/<stage>',
# which is how the figure and the tables read them back.
BLOCKS = (("self", "context"), ("nvs", "test"))
STAGES = ("input", "refined")
METRICS = ("psnr", "ssim", "lpips")

# How each metric is written out, and how many digits of it are worth
# reading: PSNR in dB to the hundredth, the other two to the thousandth.
_DIGITS = {"psnr": 2, "ssim": 4, "lpips": 4}

# Below this, two runs of the same protocol are not telling us apart:
# a validation pass of this project oscillates by about a decibel from
# one pass to the next, so a row that wins by less than this has not won.
_PSNR_NOISE_BAND = 1.0


@dataclass
class Variant:
    """
    One row of the ablation table: a name, the weights behind it, and
    whether it is also run over the whole protocol grid.
    """
    name: str
    checkpoint: str
    sweep: bool = False

    @property
    def slug(self) -> str:
        """The name as it goes into a wandb key or a file name."""
        kept = "".join(
            char if char.isalnum() else "-" for char in self.name.lower()
        )
        return "-".join(word for word in kept.split("-") if word)


def load_scene(
    cfg,
    dataset: DL3DVDataset,
    scene_idx: int,
    device: torch.device,
    num_context_views: int,
    generator: Optional[Generator] = None,
):
    """
    Sample a run of views from one scene and read just those frames,
    returning the scene alongside the two halves it splits into.
    """
    frames, context_idx, test_idx = plan_context_views(
        dataset.num_frames(scene_idx), num_context_views,
        stride=cfg.view_stride, generator=generator,
    )
    scene = dataset.get_frames(scene_idx, frames)
    scene = {name: value.to(device) for name, value in scene.items()}
    return scene, context_idx, test_idx


def score(rendered: Tensor, target: Tensor, perceptual: LPIPS) -> Dict[str, float]:
    """
    The three metrics of one render against its ground truth, each
    averaged over the views.
    """
    return {
        "psnr": psnr(rendered, target).mean().item(),
        "ssim": ssim(rendered, target).mean().item(),
        "lpips": perceptual(rendered, target).mean().item(),
    }


def build_grid(cfg, variant: Variant) -> Dict[int, List[int]]:
    """
    The cells one variant is scored at, as the budgets to sweep per
    context view count: the whole protocol grid when the variant sweeps,
    and the single ablation cell when it does not.
    """
    if not variant.sweep:
        return {cfg.ablation.context_views: [cfg.ablation.gaussian_budget]}
    return {
        num_context_views: sorted(cfg.protocol.gaussian_budgets)
        for num_context_views in cfg.protocol.context_views
    }


@torch.no_grad()
def evaluate(
    cfg,
    splatformer: SplatFormer,
    reconstructor,
    dataset: DL3DVDataset,
    scenes: Sequence[int],
    perceptual: LPIPS,
    grid: Mapping[int, Sequence[int]],
    desc: str,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Score one variant over the cells of 'grid' and return, per cell, the
    mean over the scenes of every number the cell measured.

    A scene is fit once per context view count and then thinned down to
    each budget in turn, so the budgets of a cell column are measured on
    exactly the same field. The views a scene is scored on, and the
    Gaussians a budget keeps, come off generators seeded by the scene
    alone, which is what makes two variants comparable: they are handed
    the same reconstruction, thinned the same way, and asked the same
    question of the same views.

    A scene that does not fit is dropped instead of ending the run: a
    cell's means are then over the scenes that did fit, and its
    'num_scenes' says how many that was. A cell no scene fit at is left
    out of the result entirely, which the figure draws as a gap and the
    tables print as a dash.
    """
    splatformer.eval()
    device = next(splatformer.parameters()).device
    gathered = defaultdict(lambda: defaultdict(list))

    for num_context_views in tqdm(
        sorted(grid), desc=desc, unit="view count", leave=False
    ):
        budgets = sorted(grid[num_context_views])
        for scene_idx in tqdm(
            scenes, desc=f"{num_context_views} context views",
            unit="scene", leave=False,
        ):
            seed = cfg.split_seed + scene_idx
            scene, context_idx, test_idx = load_scene(
                cfg, dataset, scene_idx, device, num_context_views,
                Generator().manual_seed(seed),
            )
            reconstruction = None
            try:
                reconstruction = reconstruct(
                    reconstructor, scene, context_idx, test_idx,
                    generator=Generator().manual_seed(seed),
                    context_downscale=cfg.context_downscale,
                )
                predicted = reconstruction.gaussians.num_gaussians
                # Cut down to the widest budget of the column as soon as
                # it is predicted. At the top of the view counts the
                # whole field is millions of Gaussians, none of which any
                # cell renders, and a budget drawn uniformly out of a
                # uniform draw is the same budget it would have drawn out
                # of the whole field.
                reconstruction.gaussians = reconstruction.gaussians.subsample(
                    budgets[-1], generator=Generator().manual_seed(seed)
                )
                # Off their own generators, seeded by the scene alone, so
                # that a scene is scored on the same views at every view
                # count and for every variant
                for half in ("context", "test"):
                    setattr(reconstruction, half, getattr(reconstruction, half).thin(
                        cfg.protocol.scored_views,
                        generator=Generator().manual_seed(seed),
                    ))
            except RuntimeError as error:
                if not out_of_memory(error):
                    raise
                error.__traceback__ = None
                reconstruction = None
                tqdm.write(
                    f"  scene {scene_idx} did not fit at {num_context_views} "
                    f"context views, skipped"
                )
            del scene
            torch.cuda.empty_cache()
            if reconstruction is None:
                continue

            halves = tuple(
                (block, getattr(reconstruction, half)) for block, half in BLOCKS
            )
            for budget in budgets:
                thinned = refined = rendered = None
                try:
                    thinned = reconstruction.gaussians.subsample(
                        budget, generator=Generator().manual_seed(seed)
                    )
                    # In single precision: spconv takes a different path
                    # once a module leaves training mode, and it has no
                    # half-precision kernel to offer there
                    with torch.cuda.amp.autocast(enabled=False):
                        refined = splatformer(thinned)
                    # Held back until every render of the cell is in, so
                    # that a cell which overflows halfway through does
                    # not leave its two stages averaged over different
                    # sets of scenes, which would put a gain between them
                    # that no scene measured
                    measured = {
                        "num_gaussians": float(thinned.num_gaussians),
                        "predicted_gaussians": float(predicted),
                    }
                    for stage, gaussians in zip(STAGES, (thinned, refined)):
                        for block, views in halves:
                            rendered, _ = gaussians.rasterize(
                                views.poses, views.intrinsics, views.image_shape,
                                views_per_pass=cfg.device_max_views_per_render,
                            )
                            measured.update({
                                f"{block}/{metric}/{stage}": value
                                for metric, value in score(
                                    rendered, views.images, perceptual
                                ).items()
                            })
                    for name, value in measured.items():
                        gathered[(num_context_views, budget)][name].append(value)
                except RuntimeError as error:
                    if not out_of_memory(error):
                        raise
                    error.__traceback__ = None
                    tqdm.write(
                        f"  scene {scene_idx} did not fit at "
                        f"{num_context_views} context views and {budget:,} "
                        f"Gaussians, skipped"
                    )
                finally:
                    del thinned, refined, rendered
                    torch.cuda.empty_cache()
            del reconstruction, halves
            torch.cuda.empty_cache()

    scores = {}
    for cell, values in gathered.items():
        means = {
            name: sum(numbers) / len(numbers) for name, numbers in values.items()
        }
        # What refinement was worth, per metric and per half, written out
        # here rather than left to every reader of the numbers to take
        # the difference themselves. It is a difference of means over one
        # set of scenes: every scene that produced one stage produced the
        # other, since a cell commits both or neither.
        means.update({
            f"{block}/{metric}/gain":
                means[f"{block}/{metric}/refined"] - means[f"{block}/{metric}/input"]
            for block, _ in BLOCKS for metric in METRICS
        })
        means["num_scenes"] = len(values["num_gaussians"])
        scores[cell] = means
    return scores


def _cell(scores, num_context_views: int, budget: int, key: str) -> Optional[float]:
    """One number of one cell, or None where nothing was measured."""
    return scores.get((num_context_views, budget), {}).get(key)


def _measured(value: Optional[float], digits: int, signed: bool = False) -> str:
    """A number as a table entry, or a dash where there is none."""
    if value is None:
        return "-"
    return f"{value:{'+' if signed else ''}.{digits}f}"


def protocol_table(
    scores,
    context_views: Sequence[int],
    budgets: Sequence[int],
    block: str = "nvs",
) -> str:
    """
    The sweep as the report's protocol frame describes it: a row per
    budget and context view count, carrying each metric of the refined
    Gaussians with what refinement won over the thinned ones beside it.

    How many scenes a row is over, and how many Gaussians it is of, go in
    with the metrics: a budget wider than the prediction is not the
    budget it asked for, and a row over fewer scenes than the others is
    one whose neighbours it cannot quite be read against.
    """
    rows = []
    for budget in budgets:
        for num_context_views in context_views:
            cell = scores.get((num_context_views, budget))
            row = [f"{budget // 1000}k", num_context_views]
            if cell is None:
                rows.append(row + ["0", "-"] + ["-"] * len(METRICS))
                continue
            row += [cell["num_scenes"], f"{cell['num_gaussians']:,.0f}"]
            row += [
                f"{_measured(cell[f'{block}/{metric}/refined'], _DIGITS[metric])} "
                f"({_measured(cell[f'{block}/{metric}/gain'], _DIGITS[metric], signed=True)})"
                for metric in METRICS
            ]
            rows.append(row)
    return tabulate(
        rows,
        headers=[
            "budget", "views", "scenes", "Gaussians",
            "PSNR ↑ (gain)", "SSIM ↑ (gain)", "LPIPS ↓ (gain)",
        ],
        colalign=("right", "right", "right", "right", "right", "right", "right"),
        tablefmt="simple",
    )


def ablation_rows(
    results: Mapping[str, dict],
    variants: Sequence[Variant],
    num_context_views: int,
    budget: int,
    block: str = "nvs",
) -> List[List[str]]:
    """
    The ablation table's body: one row per variant, each carrying the
    three metrics of the refined Gaussians at the single cell the table
    is read off.
    """
    rows = []
    for variant in variants:
        scores = results.get(variant.name, {})
        cell = scores.get((num_context_views, budget), {})
        rows.append(
            [variant.name]
            + [
                _measured(cell.get(f"{block}/{metric}/refined"), _DIGITS[metric])
                for metric in METRICS
            ]
            + [str(cell.get("num_scenes", 0))]
        )
    return rows


def ablation_latex(rows: Sequence[Sequence[str]]) -> str:
    """
    The same body written as the rows of the report's tabular, ready to
    be pasted over the placeholders in it.
    """
    width = max(len(row[0]) for row in rows)
    return "\n".join(
        f"{row[0]:<{width}} & " + " & ".join(row[1:len(METRICS) + 1]) + r" \\"
        for row in rows
    )


def log_variant(variant: Variant, scores) -> None:
    """
    Send one variant's cells to wandb as they come in, so that a sweep
    that takes hours is readable while it is still running.
    """
    wandb.log({
        f"eval/{variant.slug}/{budget // 1000}k/{num_context_views}v/{name}": value
        for (num_context_views, budget), cell in scores.items()
        for name, value in cell.items()
    })


def log_summary(results: Mapping[str, dict], variants: Sequence[Variant]) -> None:
    """
    Send the sweep to wandb as one table of every cell it measured, and
    as a line per budget for each metric, which is the chart the figure
    is drawn as here.
    """
    columns = [
        "variant", "context_views", "budget", "scenes", "gaussians",
        "block", "stage",
    ] + list(METRICS)
    table = wandb.Table(columns=columns)
    for variant in variants:
        for (num_context_views, budget), cell in sorted(
            results.get(variant.name, {}).items()
        ):
            for block, _ in BLOCKS:
                for stage in STAGES:
                    table.add_data(
                        variant.name, num_context_views, budget,
                        cell["num_scenes"], round(cell["num_gaussians"]),
                        block, stage,
                        *(cell[f"{block}/{metric}/{stage}"] for metric in METRICS),
                    )
    logged = {"eval/results": table}

    for variant in variants:
        if not variant.sweep:
            continue
        scores = results.get(variant.name, {})
        budgets = sorted({budget for _, budget in scores})
        # Only the view counts every budget was measured at: a line
        # series has one x axis for all of its lines, and a cell that
        # went missing would otherwise slide every line after it along
        shared = sorted({
            num_context_views for num_context_views, _ in scores
            if all((num_context_views, budget) in scores for budget in budgets)
        })
        if not shared:
            continue
        for block, _ in BLOCKS:
            for metric in METRICS:
                logged[f"eval/{variant.slug}/{block}/{metric}"] = wandb.plot.line_series(
                    xs=shared,
                    ys=[
                        [scores[(views, budget)][f"{block}/{metric}/refined"]
                         for views in shared]
                        for budget in budgets
                    ],
                    keys=[f"{budget // 1000}k" for budget in budgets],
                    title=f"{variant.name}: {metric} on {block} views",
                    xname="context views",
                )
    wandb.log(logged)


def write_results(path: Path, results: Mapping[str, dict], cfg) -> None:
    """
    Write every number measured so far to one JSON file, rewritten after
    each variant so that a run that is killed halfway still leaves what
    it had measured behind.
    """
    path.write_text(json.dumps({
        "config": OmegaConf.to_container(cfg, resolve=True),
        "results": {
            name: {
                f"{num_context_views}v/{budget}": cell
                for (num_context_views, budget), cell in scores.items()
            }
            for name, scores in results.items()
        },
    }, indent=2, sort_keys=True))


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    # The whole configuration a run used, resolved, at the top of its
    # log: an evaluation is read long after the overrides that made it
    # have scrolled away
    print(OmegaConf.to_yaml(cfg))
    assert cfg.reconstructor in RECONSTRUCTORS, (
        f"The reconstructor has to be one of {RECONSTRUCTORS}: "
        f"{cfg.reconstructor} is not"
    )
    assert cfg.split in cfg.splits, (
        f"The split to report on has to be one of {list(cfg.splits)}: "
        f"{cfg.split} is not"
    )
    variants = [
        Variant(**OmegaConf.to_container(variant, resolve=True))
        for variant in cfg.variants
    ]
    assert variants, "There is nothing to evaluate: no variant is configured"
    # The ablation cell is scored for every variant, including the ones
    # that sweep, so it has to be a cell of the sweep as well or the
    # table's own row would be measured somewhere the curves never go
    assert cfg.ablation.context_views in cfg.protocol.context_views, (
        f"The ablation cell is read off the sweep, so its "
        f"{cfg.ablation.context_views} context views have to be one of "
        f"{list(cfg.protocol.context_views)}"
    )
    assert cfg.ablation.gaussian_budget in cfg.protocol.gaussian_budgets, (
        f"The ablation cell is read off the sweep, so its "
        f"{cfg.ablation.gaussian_budget:,} Gaussians have to be one of "
        + ", ".join(f"{budget:,}" for budget in cfg.protocol.gaussian_budgets)
    )
    missing = [
        variant.name for variant in variants
        if not Path(variant.checkpoint).exists()
    ]
    assert not missing, (
        "Every variant is a checkpoint on this machine, and "
        + ", ".join(repr(name) for name in missing)
        + " is not there yet. Point configs/eval.yaml at the runs that "
        "produced them, or drop the rows that have not been trained."
    )

    load_dotenv()
    set_rng_seed(cfg.seed, deterministic=cfg.deterministic)
    # TF32 matmuls, which every model here is happy with and none of them
    # would run at a sensible speed without
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print("Initializing dataset...")
    dataset = DL3DVDataset(cfg.dl3dv_root_dir, cfg.dl3dv_images_subdir)
    splits = split_scenes(len(dataset), cfg.splits, cfg.split_seed)
    scenes = splits[cfg.split]
    if cfg.num_scenes is not None:
        scenes = scenes[:cfg.num_scenes]
    print(
        f"DL3DV initialized with {len(dataset)} scenes: "
        + ", ".join(f"{len(held)} {name}" for name, held in splits.items())
        + f", reporting on {len(scenes)} of the {cfg.split} split."
    )
    print(
        f"Scoring {len(variants)} variant(s) from "
        + ", ".join(str(views) for views in cfg.protocol.context_views)
        + " context views at "
        + ", ".join(
            f"{budget // 1000}k" for budget in cfg.protocol.gaussian_budgets
        )
        + f" Gaussians, on at most {cfg.protocol.scored_views} views of each "
        f"half of a scene, with the ablation table read at "
        f"{cfg.ablation.gaussian_budget // 1000}k Gaussians and "
        f"{cfg.ablation.context_views} context views."
    )

    print(f"Initializing frozen {cfg.reconstructor} from pre-trained checkpoint...")
    reconstructor = build_reconstructor(
        cfg.reconstructor, cfg.anysplat_checkpoint, cfg.yonosplat_checkpoint,
    ).to(device)
    # One VGG for the whole run, built before the first variant is: it is
    # the same network every cell is scored through
    perceptual = LPIPS()

    run = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=cfg.wandb.run_name,
        notes=cfg.wandb.notes,
        job_type="eval",
        settings=wandb.Settings(x_disable_stats=not cfg.wandb.system_metrics),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    output_dir = Path(cfg.log.output_dir) / (
        run.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {output_dir}")

    results: Dict[str, dict] = {}
    for variant in tqdm(variants, desc="Evaluating", unit="variant"):
        tqdm.write(f"Scoring {variant.name} from {variant.checkpoint}...")
        # One SplatFormer on the card at a time: a variant is a few
        # hundred megabytes of weights, and they never have to meet
        splatformer = SplatFormer(str(variant.checkpoint), quiet=True).to(device)
        results[variant.name] = evaluate(
            cfg, splatformer, reconstructor, dataset, scenes, perceptual,
            build_grid(cfg, variant), desc=variant.name,
        )
        del splatformer
        gc.collect()
        torch.cuda.empty_cache()
        log_variant(variant, results[variant.name])
        write_results(output_dir / "results.json", results, cfg)
        cell = results[variant.name].get(
            (cfg.ablation.context_views, cfg.ablation.gaussian_budget), {}
        )
        if cell:
            tqdm.write(
                f"  {variant.name} at {cfg.ablation.gaussian_budget // 1000}k "
                f"Gaussians and {cfg.ablation.context_views} context views: "
                + ", ".join(
                    f"{metric.upper()} "
                    f"{cell[f'nvs/{metric}/refined']:.{_DIGITS[metric]}f} "
                    f"({cell[f'nvs/{metric}/gain']:+.{_DIGITS[metric]}f})"
                    for metric in METRICS
                )
                + f" over {cell['num_scenes']} scenes"
            )

    figures = {}
    for variant in variants:
        if not variant.sweep:
            continue
        figure = plot_eval_curves(
            results[variant.name],
            list(cfg.protocol.context_views),
            sorted(cfg.protocol.gaussian_budgets),
            title=(
                f"{variant.name} on the DL3DV {cfg.split} split, "
                f"{len(scenes)} scenes through {cfg.reconstructor}"
            ),
        )
        # The name the report's figure carries, with the variant in it
        # only once there is more than one sweep to tell apart
        suffix = "" if sum(one.sweep for one in variants) == 1 else f"_{variant.slug}"
        path = output_dir / f"fig_eval_curves{suffix}.png"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        figures[f"eval/{variant.slug}/curves"] = wandb.Image(figure)
        plt.close(figure)
        print(f"Wrote {path}")

    log_summary(results, variants)
    if figures:
        wandb.log(figures)

    # Everything the report asks for, printed where the run can be read
    # off the terminal rather than out of wandb
    for variant in variants:
        if not variant.sweep:
            continue
        for block, _ in BLOCKS:
            print(
                f"\n{variant.name}, "
                + ("self-reconstruction, on the views the reconstructor saw"
                   if block == "self"
                   else "novel view synthesis, on the views held out from it")
                + f", over the DL3DV {cfg.split} split\n"
            )
            print(protocol_table(
                results[variant.name],
                list(cfg.protocol.context_views),
                sorted(cfg.protocol.gaussian_budgets),
                block=block,
            ))

    rows = ablation_rows(
        results, variants, cfg.ablation.context_views,
        cfg.ablation.gaussian_budget,
    )
    print(
        f"\nAblation, novel view synthesis at "
        f"{cfg.ablation.gaussian_budget // 1000}k Gaussians and "
        f"{cfg.ablation.context_views} context views\n"
    )
    print(tabulate(
        rows,
        headers=["variant", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "scenes"],
        colalign=("left", "right", "right", "right", "right"),
        tablefmt="simple",
    ))
    print(
        f"\nA pass of this protocol oscillates by about "
        f"{_PSNR_NOISE_BAND:.0f} dB from one run to the next, so two rows "
        f"inside that band are not separated by their PSNR."
    )

    latex = ablation_latex(rows)
    (output_dir / "ablation.tex").write_text(latex + "\n")
    print("\nThe same rows, for the report's tabular:\n")
    print(latex)
    print(f"\nWrote {output_dir / 'ablation.tex'}")

    wandb.finish()


if __name__ == "__main__":
    main()
