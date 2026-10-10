"""
Figures and tables of a scripts/evaluate_all.py run.

    python scripts/plot_evaluate_all.py outputs/evaluate-all [--out DIR] [--gated]

For every (dataset, reconstructor) it writes, into DIR (default <run>/figures):
    <dataset>-<reconstructor>-nvs.png / -self.png
        the layout of every fraction figure so far: rows PSNR / SSIM / LPIPS, one column per view count,
        x the share of the predicted field kept; every score, the uniform draw and AnyPrune
    tables.md
        per metric and view count: AnyPrune, the best other score at that share (chosen per view count,
        share and metric, by mean over scenes) and the uniform draw, with the paired difference between
        AnyPrune and that score (mean +- SEM over scenes); the unpruned and whole refined field; the own cut
    results.json
        the aggregate of evaluate_all.aggregate, with the derived curves when asked for

--main RECONSTRUCTOR=NAME makes NAME the AnyPrune model drawn and tabulated as "AnyPrune" on that reconstructor's
fields (default: the first name in alphabetical order); the other models are drawn as one dashed curve each (their
refined cut), labelled by --label NAME=TEXT. --models NAME,... draws only those; --datasets / --reconstructors filter.

--gated adds <name>-gated per AnyPrune model: at every share, the refined cut or the unrefined one
(<name>-logits), whichever renders the context views better. It needs nothing but the context views,
so it is available at inference; it is derived from the records already in the cells.
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

import evaluate_all as ea
import plot_fractions as pf
from anyprune.viz import plot_fraction_curves

FULL = ea.FULL
METRICS = (("psnr", "PSNR (dB) ↑", True), ("ssim", "SSIM ↑", True), ("lpips", "LPIPS ↓", False))
TABLE_FRACTIONS = (0.1, 0.2, 0.3, 0.5, 0.7)
SHORT = {"uniform": "Uniform", "radsplat": "RadSplat", "speedy-splat": "Speedy", "pup": "PUP",
         "mini-splatting": "Mini", "lightgaussian": "LightG", "refine": "REFINE", "gaussianpop": "GPOP"}


def anyprune_names(cells) -> list:
    return sorted({r["method"][:-len("-own")] for c in cells for r in c["records"] if r["method"].endswith("-own")})


def add_gated(cells, names):
    """<name>-gated: at each share the refined or the unrefined cut, whichever renders the context views better."""
    for cell in cells:
        index = {(r["method"], round(r["fraction"], 4)): r for r in cell["records"]}
        for name in names:
            for (method, fraction), refined in list(index.items()):
                if method != name:
                    continue
                plain = index.get((f"{name}-logits", fraction)) if fraction < 1 else index.get((FULL, 1.0))
                if plain is None:
                    continue
                chosen = refined if refined["self_psnr"] >= plain["self_psnr"] else plain
                cell["records"].append({**chosen, "method": f"{name}-gated", "fraction": refined["fraction"]})


def styles(methods, names, labels=None, others_dashed=False):
    """
    Every method's (label, colour, marker, linestyle), in plot_fractions' colours, AnyPrune models after. With
    'others_dashed', every model after the first is one dashed curve (its refined cut), labelled from 'labels'.
    """
    labels = labels or {}
    out = {}
    for method in pf.METHODS:
        if method in methods and not method.startswith("anyprune"):
            out[method] = pf.METHODS[method]
    shades = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]
    dashed = ["#8c564b", "#7570b3", "#e7298a", "#17becf"]
    for i, name in enumerate(names):
        color = pf.METHODS["anyprune"][1] if i == 0 else shades[i % len(shades)]
        if others_dashed and i > 0:
            if name in methods:
                out[name] = (labels.get(name, name), dashed[(i - 1) % len(dashed)], "v", "--")
            continue
        if name.endswith("-zeroshot"):
            # The YoNoSplat-trained model on another reconstructor's fields: only its refined curve is drawn
            if name in methods:
                out[name] = ("AnyPrune zero-shot (trained on YoNoSplat)", "#8c564b", "v", "--")
            continue
        label = labels.get(name, "AnyPrune" if name == "anyprune" or (others_dashed and i == 0) else name)
        variants = {name: (f"{label} (mask logits + refiner)", color, "h", "-"),
                    f"{name}-logits": (f"{label} logits, raw", color, "h", ":"),
                    f"{name}-whole": (f"{label}, whole residual", "#8fd6bb", "h", "--"),
                    f"{name}-gated": (f"{label}, refiner gated on context views", "#0b6b4a", "p", "-."),
                    f"{name}-own": (f"{label} own cut (logit > 0)", "#0b6b4a", "*", "none")}
        out.update({m: s for m, s in variants.items() if m in methods})
    return out


def figure_summary(summary, names):
    """The aggregate as the figure draws it: every raw curve ends on the predicted field, the refined ones on their own."""
    drawn = []
    refined_ends = set(names) | {f"{n}-whole" for n in names} | {f"{n}-gated" for n in names}
    methods = {r["method"] for r in summary}
    for r in summary:
        if r["method"] == FULL:
            drawn += [{**r, "method": m} for m in methods if m != FULL and m not in refined_ends and not r["own_cut"]
                      and not m.endswith("-own")]
        else:
            drawn.append(r)
    return drawn


def paired(cells, a, b, fraction, key):
    """Mean and SEM over scenes of a - b at one share."""
    diffs = []
    for cell in cells:
        index = {(r["method"], round(r["fraction"], 4)): r for r in cell["records"]}
        ra = index.get((a, fraction)) or (index.get((FULL, 1.0)) if fraction == 1.0 else None)
        rb = index.get((b, fraction)) or (index.get((FULL, 1.0)) if fraction == 1.0 else None)
        if ra is not None and rb is not None:
            diffs.append(ra[key] - rb[key])
    if len(diffs) < 2:
        return float("nan"), float("nan")
    mean = sum(diffs) / len(diffs)
    sem = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs))
    return mean, sem


def tables(cells, summary, dataset, reconstructor, names, label) -> str:
    lines = [f"\n## {label}, {reconstructor}\n"]
    rows = [r for r in summary if r["dataset"] == dataset and r["reconstructor"] == reconstructor]
    views = sorted({r["context_views"] for r in rows})
    main = names[0] if names else None
    zeroshot = next((n for n in names if n.endswith("-zeroshot") and any(r["method"] == n for r in rows)), None)
    for (block, block_title), (metric, title, higher) in [(b, m) for b in (("nvs", "Novel views"),
                                                                           ("self", "Context views (self-reconstruction)"))
                                                          for m in METRICS]:
        key = f"{block}_{metric}"
        fmt = "{:.2f}" if metric == "psnr" else "{:.3f}"
        lines.append(f"\n**{block_title}, {title}**: {main or 'AnyPrune'} / best other score (name) / uniform; "
                     f"below, {main} − best other, mean ± SEM over scenes"
                     + ("; then the YoNoSplat-trained model zero-shot" if zeroshot else "") + "\n")
        lines.append("| views (scenes, Gaussians) | unpruned | " + " | ".join(f"{f:.0%}" for f in TABLE_FRACTIONS)
                     + " | own cut (kept) |")
        lines.append("|---|---|" + "---|" * len(TABLE_FRACTIONS) + "---|")
        for v in views:
            at = {(r["method"], None if r["own_cut"] else round(r["fraction"], 4)): r
                  for r in rows if r["context_views"] == v}
            full = at.get((FULL, 1.0))
            if full is None:
                continue
            group = [c for c in cells if c["dataset"] == dataset and c["reconstructor"] == reconstructor
                     and c["context_views"] == v]
            unpruned = fmt.format(full[key])
            if main and (main, 1.0) in at:
                unpruned += f" (refined {fmt.format(at[(main, 1.0)][key])})"
            row = [f"{v} ({full['num_scenes']}, {full['num_gaussians'] / 1e3:,.0f}k)", unpruned]
            for f in TABLE_FRACTIONS:
                others = {m: r[key] for (m, fr), r in at.items() if fr == f and m in SHORT and m != "uniform"}
                if not others:
                    row.append("–")
                    continue
                best = (max if higher else min)(others, key=others.get)
                uniform = at.get(("uniform", f))
                text = f"{fmt.format(others[best])} ({SHORT[best]})" + (f" / {fmt.format(uniform[key])}" if uniform else "")
                if main and (main, f) in at:
                    ours = at[(main, f)][key]
                    win = ours > others[best] if higher else ours < others[best]
                    mean, sem = paired(group, main, best, f, key)
                    text = (f"{'**' if win else ''}{fmt.format(ours)}{'**' if win else ''} / {text}"
                            f"<br>{mean:+.{2 if metric == 'psnr' else 3}f} ± {sem:.{2 if metric == 'psnr' else 3}f}")
                    if zeroshot and (zeroshot, f) in at:
                        text += f"<br>zero-shot {fmt.format(at[(zeroshot, f)][key])}"
                row.append(text)
            own = at.get((f"{main}-own", None)) if main else None
            row.append(f"{fmt.format(own[key])} ({own['fraction']:.0%})" if own else "–")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--gated", action="store_true", help="add the context-view refiner gate as a derived curve")
    parser.add_argument("--main", action="append", default=[], help="RECONSTRUCTOR=NAME, the model drawn as AnyPrune there")
    parser.add_argument("--label", action="append", default=[], help="NAME=TEXT, a model's legend label")
    parser.add_argument("--models", default=None, help="comma-separated: draw only these AnyPrune models")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--reconstructors", default=None)
    args = parser.parse_args()
    mains = dict(m.split("=", 1) for m in args.main)
    labels = dict(m.split("=", 1) for m in args.label)
    keep = None if args.models is None else args.models.split(",")
    out = args.out or args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cells = ea.read_cells(args.run)
    assert cells, f"No cells under {args.run / 'cells'}"
    if args.datasets:
        cells = [c for c in cells if c["dataset"] in args.datasets.split(",")]
    if args.reconstructors:
        cells = [c for c in cells if c["reconstructor"] in args.reconstructors.split(",")]
    names = anyprune_names(cells)
    if keep is not None:
        hidden = set(names) - set(keep)
        hidden_methods = {f"{n}{v}" for n in hidden for v in ("", "-own", "-logits", "-whole", "-gated")}
        for cell in cells:
            cell["records"] = [r for r in cell["records"] if r["method"] not in hidden_methods]
        names = [n for n in names if n in keep]
    if args.gated:
        add_gated(cells, names)
    summary = ea.aggregate(cells)
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    config = ea.OmegaConf.load(args.run / "config.yaml") if (args.run / "config.yaml").exists() else None
    report = [f"# {args.run.name}\n\nNovel-view and context-view metrics. Bold: AnyPrune beats the best other score at that share. "
              "The paired difference is over the scenes both have."]
    for dataset, reconstructor in sorted({(c["dataset"], c["reconstructor"]) for c in cells}):
        label = config.datasets[dataset].label if config is not None and dataset in config.datasets else dataset
        rows = [r for r in summary if r["dataset"] == dataset and r["reconstructor"] == reconstructor]
        views = sorted({r["context_views"] for r in rows})
        ordered = names
        if reconstructor in mains:
            ordered = [mains[reconstructor]] + [n for n in names if n != mains[reconstructor]]
        methods = styles({r["method"] for r in rows}, ordered, labels, others_dashed=bool(mains))
        scenes = max(r["num_scenes"] for r in rows)
        for block, block_title in pf.BLOCKS.items():
            figure = plot_fraction_curves(
                figure_summary(rows, names), block, views, methods,
                f"{block_title.capitalize()}, {label}, {reconstructor} (up to {scenes} scenes)",
                metrics=[(m, t) for m, t, _ in METRICS], reference="uniform",
            )
            figure.savefig(out / f"{dataset}-{reconstructor}-{block}.png", dpi=110, bbox_inches="tight")
            plt.close(figure)
        report.append(tables(cells, summary, dataset, reconstructor, ordered, label))
    (out / "tables.md").write_text("\n".join(report) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
