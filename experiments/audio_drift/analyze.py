"""Offline layer/progress heatmaps and intervention curves from scalar metrics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:60] + "_" + hashlib.sha256(name.encode()).hexdigest()[:8]


def collect(run, bins, groups, references, tensors):
    # Each sample contributes equally after averaging steps within its bin.
    cells = defaultdict(list)
    for sample in sorted((run / "samples").iterdir()):
        if not (sample / "result.json").exists() or sample.name.startswith("."):
            continue
        task = json.loads((sample / "result.json").read_text())["task"]
        within = defaultdict(list)
        with (sample / "drift.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["age"] == 0 or row["group"] not in groups or row["reference"] not in references or row["tensor"] not in tensors:
                    continue
                bin_index = min(bins - 1, int((1 - row["remaining_fraction"]) * bins))
                for metric in ("relative_l2", "cosine_distance"):
                    key = (task, row["group"], row["reference"], row["tensor"], metric,
                           row["layer"], bin_index)
                    within[key].append(row["metrics"][metric]["median"])
        for key, values in within.items():
            cells[key].append(float(np.mean(values)))
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--groups", nargs="+", default=["audio_semantic", "audio_acoustic", "prompt", "answer_decoded"])
    p.add_argument("--references", nargs="+", default=["previous", "block_start"])
    p.add_argument("--tensors", nargs="+", default=["hidden", "key", "value"])
    args = p.parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.run / "analysis"
    output.mkdir(exist_ok=True)
    cells = collect(args.run, args.bins, args.groups, args.references, args.tensors)
    panels = defaultdict(dict)
    rows = []
    for key, values in sorted(cells.items()):
        panel, layer, bin_index = key[:-2], key[-2], key[-1]
        panels[panel][layer, bin_index] = float(np.mean(values))
        rows.append({"task": key[0], "group": key[1], "reference": key[2], "tensor": key[3],
                     "metric": key[4], "layer": layer, "progress_bin": bin_index,
                     "sample_count": len(values), "mean_sample_median": float(np.mean(values))})
    (output / "heatmap_values.json").write_text(json.dumps(rows, indent=2) + "\n")
    shared_max = defaultdict(float)
    for panel, values in panels.items():
        if panel[1] in {"audio_semantic", "audio_acoustic", "prompt"}:
            scale_key = panel[0], *panel[2:]
            shared_max[scale_key] = max(shared_max[scale_key], max(values.values()))
    for panel, values in panels.items():
        layers = sorted({layer for layer, _ in values})
        matrix = np.full((len(layers), args.bins), np.nan)
        for i, layer in enumerate(layers):
            for b in range(args.bins):
                matrix[i, b] = values.get((layer, b), np.nan)
        fig, ax = plt.subplots(figsize=(9, 6))
        scale = shared_max.get((panel[0], *panel[2:])) if panel[1] in {"audio_semantic", "audio_acoustic", "prompt"} else None
        image = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", vmin=0,
                          vmax=max(scale, 1e-8) if scale is not None else None)
        ax.set_yticks(range(len(layers)), layers)
        ax.set_xticks(range(args.bins), [f"{1 - (b + .5) / args.bins:.2f}" for b in range(args.bins)], rotation=45)
        ax.set(xlabel="Fraction of answer masks remaining (bin centers)", ylabel="Transformer layer (0 based)",
               title=" / ".join(panel) + "\nMean of per-sample, per-bin token medians")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / (safe_name("_".join(panel)) + ".png"), dpi=150)
        plt.close(fig)
    curves = defaultdict(lambda: defaultdict(list))
    paired_baselines = defaultdict(dict)
    for path in sorted((args.run / "samples").glob("*/result.json")):
        result = json.loads(path.read_text())
        if path.parent.name.startswith("."):
            continue
        for branch in result["interventions"]:
            score = branch["score"]
            if score is None:
                continue
            metric = "accuracy" if "correct" in score else "wer"
            value = float(score["correct"]) if metric == "accuracy" else score["wer"]
            baseline_score = result["score"]
            baseline = float(baseline_score["correct"]) if metric == "accuracy" else baseline_score["wer"]
            schedules = branch.get("schedule", [])
            if branch["mode"] != "freeze":
                panel = "remove / text-only"
            elif len(schedules) > 1:
                panel = "mixed schedules"
            else:
                panel = schedules[0]["group"] if schedules else "both"
            for subset in ["all", "unfinished"]:
                if subset == "unfinished" and branch["baseline_answer_already_revealed"]:
                    continue
                key = result["task"], metric, subset, panel
                condition = branch.get("condition", branch["mode"])
                fraction = branch["anchor_step_fraction"]
                curves[key][condition, fraction].append(value)
                paired_baselines[key][fraction, result["id"]] = baseline
    figures = defaultdict(list)
    for key in curves:
        figures[key[:3]].append(key)
    for (task, metric, subset), keys in figures.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False, layout="constrained")
        for ax, key in zip(axes.flat, sorted(keys)):
            values = curves[key]
            for mode in sorted({k[0] for k in values}):
                fractions = sorted({k[1] for k in values if k[0] == mode})
                ax.plot(fractions, [np.mean(values[mode, f]) for f in fractions], marker="o", label=mode)
            pairs = paired_baselines[key]
            fractions = sorted({f for f, _ in pairs})
            ax.plot(fractions, [np.mean([v for (f, _), v in pairs.items() if f == fraction]) for fraction in fractions],
                    color="black", linestyle="--", label="paired baseline")
            ax.set(xlabel="Baseline anchor fraction", ylabel="Mean sample " + metric,
                   title=key[3])
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        fig.suptitle(f"{task}: {subset} baseline answers; sequential paired continuations")
        fig.savefig(output / (safe_name(task + "_interventions_" + metric + "_" + subset) + ".png"), dpi=150)
        plt.close(fig)
    print(f"Wrote {len(panels)} heatmaps and {len(figures)} intervention plots to {output}")


if __name__ == "__main__":
    main()
