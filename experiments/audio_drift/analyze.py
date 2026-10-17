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
    for panel, values in panels.items():
        layers = sorted({layer for layer, _ in values})
        matrix = np.full((len(layers), args.bins), np.nan)
        for i, layer in enumerate(layers):
            for b in range(args.bins):
                matrix[i, b] = values.get((layer, b), np.nan)
        fig, ax = plt.subplots(figsize=(9, 6))
        image = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", vmin=0)
        ax.set_yticks(range(len(layers)), layers)
        ax.set_xticks(range(args.bins), [f"{1 - (b + .5) / args.bins:.2f}" for b in range(args.bins)], rotation=45)
        ax.set(xlabel="Fraction of answer masks remaining (bin centers)", ylabel="Transformer layer (0 based)",
               title=" / ".join(panel) + "\nMean of per-sample, per-bin token medians")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / (safe_name("_".join(panel)) + ".png"), dpi=150)
        plt.close(fig)
    curves = defaultdict(lambda: defaultdict(list))
    for path in sorted((args.run / "samples").glob("*/result.json")):
        result = json.loads(path.read_text())
        if path.parent.name.startswith("."):
            continue
        for branch in result["interventions"]:
            for condition, score in ((branch["mode"], branch["score"]), ("baseline", result["score"])):
                if score is None:
                    continue
                metric = "accuracy" if "correct" in score else "wer"
                value = float(score["correct"]) if metric == "accuracy" else score["wer"]
                curves[result["task"], metric][condition, branch["anchor_step_fraction"]].append(value)
    for (task, metric), values in curves.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode in sorted({k[0] for k in values}):
            fractions = sorted({k[1] for k in values if k[0] == mode})
            ax.plot(fractions, [np.mean(values[mode, f]) for f in fractions], marker="o", label=mode)
        ax.set(xlabel="Anchor fraction of baseline forward calls", ylabel="Mean sample " + metric,
               title=task + ": continued audio dependence")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / (safe_name(task + "_interventions_" + metric) + ".png"), dpi=150)
        plt.close(fig)
    print(f"Wrote {len(panels)} heatmaps and {len(curves)} intervention plots to {output}")


if __name__ == "__main__":
    main()
