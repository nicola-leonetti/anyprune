"""
Aggregate and draw a sweep of the pruning scores over kept fractions:
every score of eval_learned.SCORES and the uniform draw, each cut to the
top 10..90% of the predicted field (raw, no compensation or refiner), on
the DL3DV test split, from one JSON per (view count, scene) under
<run>/cells/<views>v/scene-<index>.json.

A cell holds 'context_views', 'scene' and one record per (method,
fraction) with PSNR, SSIM and LPIPS on both halves of the scene
('self_<metric>' on the context views, 'nvs_<metric>' on the test
views); the whole field is the one record of method "full field" at
1.0, which every method's curve ends on.

    python scripts/plot_fractions.py outputs/eval-fractions/pruning-scores

AnyPrune's cells (the refined field cut by its mask logits, the logits
as a raw score, and its own logit > 0 cut) are measured on the same
fields in a run of their own and merged in by (views, scene):

    python scripts/plot_fractions.py outputs/eval-fractions/pruning-scores \
        --with outputs/eval-fractions/anyprune-nodeg --out outputs/eval-fractions/with-anyprune-nodeg

Writes into the run directory, or --out:
    results.json    the mean, std and SEM over scenes of every
                    (views, method, fraction, metric), with every cell
    nvs.png         test views: rows PSNR/SSIM/LPIPS, columns the view
                    counts, x the share of Gaussians kept
    self.png        the same on the context views
    nvs_pareto.png, self_pareto.png
                    the same with only the methods no other one
                    dominates on that figure (see dominance()), and
                    the uniform draw as the reference
    pareto.json     which methods dominate which, per figure
    report.md       the tables
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_learned as E
from anyprune.viz import REFERENCE_COLOR, SERIES_COLORS, plot_fraction_curves

FULL = "full field"
# What a merged run scores the whole field again as, to check that its
# fields are the main run's; never drawn
FULL_CHECK = "full field check"
REFERENCE = "uniform"
# Every method in the order of the legend: (label, colour, marker,
# linestyle), each score in the colour eval_learned gives it
MARKERS = {
    "speedy-splat": "o", "gaussianpop": "P", "radsplat": "s", "mini-splatting": "v",
    "refine": "X", "pup": "D", "lightgaussian": "^",
}
# AnyPrune in the colour eval_learned gives the learned mask. Its
# refined curve ends on the whole refined field, not the predicted one;
# its own rule is one point per scene, at whatever share it kept, drawn
# as one marker at the mean share
ANYPRUNE, ANYPRUNE_OWN = "anyprune", "anyprune-own"
METHODS = {
    ANYPRUNE: ("AnyPrune (mask logits + refiner)", SERIES_COLORS[2], "h", "-"),
    "anyprune-logits": ("AnyPrune logits, raw", SERIES_COLORS[2], "h", ":"),
    ANYPRUNE_OWN: ("AnyPrune own rule (logit > 0)", "#0b6b4a", "*", "none"),
    # The same forward with the refiner's whole residual, next to a curve
    # that scales it down by the field's size (--also-full)
    "anyprune-whole": ("AnyPrune, whole residual", "#8fd6bb", "h", "--"),
    # The model without the LPIPS term, next to one trained with it
    "anyprune-nolpips": ("AnyPrune without LPIPS (current final)", "#0b6b4a", "h", "--"),
    **{score: (*E.SCORES[score], marker, "-") for score, marker in MARKERS.items()},
    REFERENCE: ("Uniform (random)", REFERENCE_COLOR, None, "--"),
}
OWN_FULL = {ANYPRUNE, "anyprune-whole", "anyprune-nolpips"}
METRICS = (("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓"))
BLOCKS = {"nvs": "novel view synthesis (test views)", "self": "self-reconstruction (context views)"}
# How many paired standard errors a difference has to clear to count
DOMINANCE_SEMS = 2.0


def load(run: Path, merged=()):
    """The cells of a run, each with the records of the same (views, scene) of every merged run added."""
    def read(directory: Path):
        return {(c["context_views"], c["scene"]): c
                for c in (json.loads(p.read_text()) for p in sorted((directory / "cells").glob("*v/scene-*.json")))}

    cells = read(run)
    assert cells, f"No cells under {run / 'cells'}"
    for other in merged:
        extra = read(other)
        assert extra, f"No cells under {other / 'cells'}"
        missing = sorted(extra.keys() - cells.keys())
        assert not missing, f"{other} has cells the main run lacks: {missing[:5]}"
        for key, cell in extra.items():
            full = next(r for r in cells[key]["records"] if r["method"] == FULL)
            check = next((r for r in cell["records"] if r["method"] == FULL_CHECK), None)
            assert check is None or all(check[k] == full[k] for k in full if k != "method"), (
                f"{other} scored another field than {run} at {key}"
            )
            cells[key]["records"] = cells[key]["records"] + [r for r in cell["records"] if r["method"] != FULL_CHECK]
    return list(cells.values())


def aggregate(cells):
    """The mean, std and SEM over scenes of every (views, method, fraction), the full field under every method at 1.0."""
    groups = defaultdict(list)
    for cell in cells:
        for r in cell["records"]:
            methods = [m for m in METHODS if m not in OWN_FULL | {ANYPRUNE_OWN}] if r["method"] == FULL else [r["method"]]
            for method in methods:
                groups[(cell["context_views"], method, None if method == ANYPRUNE_OWN else r["fraction"])].append(r)
    summary = []
    for (views, method, fraction), group in sorted(groups.items(), key=lambda g: (g[0][0], g[0][1], g[0][2] or 0.0)):
        if fraction is None:
            fraction = float(np.mean([r["fraction"] for r in group]))
        row = {"context_views": views, "method": method, "fraction": fraction, "num_scenes": len(group),
               "num_gaussians": float(np.mean([r["num_gaussians"] for r in group]))}
        for block in BLOCKS:
            for metric, _ in METRICS:
                values = np.array([r[f"{block}_{metric}"] for r in group])
                row[f"{block}_{metric}"] = float(values.mean())
                row[f"{block}_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"{block}_{metric}_sem"] = row[f"{block}_{metric}_std"] / math.sqrt(len(values))
        summary.append(row)
    return summary


def dominance(cells, block: str, views_list) -> dict:
    """
    For every method, the methods that dominate it on one figure: over
    every view count, metric and fraction under 1.0 of it, never worse
    by more than DOMINANCE_SEMS standard errors of the per-scene
    difference, and better by more than that at one of them at least.
    Differences are paired by scene, since a scene's difficulty moves
    every method together.
    """
    values = defaultdict(dict)
    for cell in cells:
        for r in cell["records"]:
            if r["method"] not in (FULL, ANYPRUNE_OWN) and r["fraction"] < 1.0:
                for metric, _ in METRICS:
                    values[(cell["context_views"], metric, r["method"], r["fraction"])][cell["scene"]] = r[f"{block}_{metric}"]
    sign = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0}
    fractions = sorted({f for (_, _, _, f) in values})

    def dominates(winner: str, loser: str) -> bool:
        better = False
        for views in views_list:
            for metric, _ in METRICS:
                for f in fractions:
                    a, b = values[(views, metric, winner, f)], values[(views, metric, loser, f)]
                    shared = sorted(a.keys() & b.keys())
                    d = sign[metric] * (np.array([a[s] for s in shared]) - np.array([b[s] for s in shared]))
                    margin = DOMINANCE_SEMS * d.std(ddof=1) / math.sqrt(len(d))
                    if d.mean() < -margin:
                        return False
                    better |= d.mean() > margin
        return better

    ranked = [m for m in METHODS if m != ANYPRUNE_OWN]
    return {loser: [w for w in ranked if w != loser and dominates(w, loser)] for loser in ranked}


def report(summary, views_list, num_scenes: str) -> str:
    lines = [f"# Pruning scores vs kept fraction, DL3DV test ({num_scenes} scenes), {E.RECONSTRUCTOR}, raw top-k\n"]
    for block, block_name in BLOCKS.items():
        for metric, label in METRICS:
            lines.append(f"\n## {block_name}: {label}\n")
            for views in views_list:
                rows = {(r["method"], r["fraction"]): r for r in summary if r["context_views"] == views}
                fractions = sorted({f for m, f in rows if m != ANYPRUNE_OWN})
                lines.append(f"\n**{views} context views**\n")
                lines.append("| method | " + " | ".join(f"{f:.0%}" for f in fractions) + " |")
                lines.append("|---" * (len(fractions) + 1) + "|")
                for method, (name, *_) in METHODS.items():
                    if method == ANYPRUNE_OWN:
                        continue
                    cells = [rows.get((method, f)) for f in fractions]
                    fmt = "{:.2f}" if metric == "psnr" else "{:.3f}"
                    lines.append(f"| {name} | " + " | ".join(
                        fmt.format(c[f"{block}_{metric}"]) if c else "" for c in cells) + " |")
                own = next((r for (m, _), r in rows.items() if m == ANYPRUNE_OWN), None)
                if own:
                    lines.append(f"\n{METHODS[ANYPRUNE_OWN][0]}: {fmt.format(own[f'{block}_{metric}'])} "
                                 f"at {own['fraction']:.1%} kept (mean over scenes)")
    return "\n".join(lines) + "\n"


def save(figure, path: Path):
    figure.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the directory holding cells/")
    parser.add_argument("--with", dest="merged", type=Path, nargs="*", default=[],
                        help="runs whose cells are measured on the same fields, merged in by (views, scene)")
    parser.add_argument("--out", type=Path, default=None, help="where to write; the run directory by default")
    parser.add_argument("--dataset", default="DL3DV test", help="what the titles name the scenes as")
    args = parser.parse_args()
    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)

    cells = load(args.run, args.merged)
    present = {r["method"] for c in cells for r in c["records"]}
    for method in [m for m in METHODS if m not in present]:
        del METHODS[method]
    summary = aggregate(cells)
    views_list = sorted({c["context_views"] for c in cells})
    scenes = {v: len({c["scene"] for c in cells if c["context_views"] == v}) for v in views_list}
    num_scenes = str(scenes[views_list[0]]) if len(set(scenes.values())) == 1 else "/".join(map(str, scenes.values()))
    (out / "results.json").write_text(json.dumps({
        "config": cells[0]["config"], "scenes_per_view_count": scenes, "summary": summary, "cells": cells,
    }, indent=1))

    pareto = {}
    for block, name in BLOCKS.items():
        title = f"{name.capitalize()}, {args.dataset} ({num_scenes} scenes)"
        save(plot_fraction_curves(summary, block, views_list, METHODS, title, METRICS, REFERENCE),
             out / f"{block}.png")
        dominated_by = dominance(cells, block, views_list)
        # AnyPrune's own point stands or falls with its curve
        if ANYPRUNE_OWN in METHODS:
            dominated_by[ANYPRUNE_OWN] = dominated_by.get(ANYPRUNE, [])
        front = {m: style for m, style in METHODS.items() if not dominated_by[m] or m == REFERENCE}
        pareto[block] = {"front": list(front), "dominated_by": dominated_by}
        save(plot_fraction_curves(summary, block, views_list, front, f"{title}: non-dominated methods", METRICS, REFERENCE),
             out / f"{block}_pareto.png")
        print(f"{block}: front {list(front)}; dropped " + ", ".join(
            f"{m} (by {', '.join(by)})" for m, by in dominated_by.items() if by and m != REFERENCE))
    (out / "pareto.json").write_text(json.dumps({
        "rule": f"A is dominated by B on a figure when, at every view count, metric and fraction < 1.0, the "
                f"per-scene paired mean of B - A (sign-flipped for LPIPS) is > -{DOMINANCE_SEMS:g} SEM, and "
                f"> +{DOMINANCE_SEMS:g} SEM at one of them at least; the uniform draw is kept as the reference",
        **pareto,
    }, indent=1))
    (out / "report.md").write_text(report(summary, views_list, num_scenes))
    print(f"Wrote results.json, nvs.png, self.png, nvs_pareto.png, self_pareto.png, pareto.json and report.md "
          f"to {out} ({scenes} scenes per view count)")


if __name__ == "__main__":
    main()
