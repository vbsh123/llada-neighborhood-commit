"""VidLaDA-style cosine figures with explicit pooling and active-answer markers."""
import argparse
import json
from pathlib import Path

import numpy as np


GROUPS = ("audio_acoustic", "audio_semantic", "prompt")


def depth_summary(matrices, layers, end):
    """Keep layer 0, exclude step-zero self-comparison, never pool missing cells."""
    if end is None or end < 1:
        return None
    report = {"layers": list(layers), "all_32_layers_observed": list(layers) == list(range(32)),
              "pooling": "Mean token cosine distance, then equal forward and layer weights; layer 0 included",
              "active_mean_all_measured_layers": {}, "active_mean_by_layer": {}}
    for group in GROUPS:
        values = matrices['previous', group][:, 1:end + 1]
        if not np.isfinite(values).all():
            raise ValueError("Missing active layer/step cells; refusing an incomplete depth average")
        means = values.mean(axis=1)
        report['active_mean_all_measured_layers'][group] = float(means.mean())
        report['active_mean_by_layer'][group] = {str(layer): float(value) for layer, value in zip(layers, means)}
    return report


def analyze(run, output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    output = Path(output) if output else run / "analysis" / "cosine"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"metric": "mean per-token cosine distance/similarity, not relative L2 percent",
               "comparison_note": "Reproduces figure types; VidLaDA's exact layer/token pooling and "
                                  "sample aggregation are not established. Step gaps and post-answer "
                                  "regions must be matched before numerical comparisons.", "samples": []}
    for path in sorted((run / "samples").glob("*/result.json")):
        if path.parent.name.startswith("."):
            continue
        result = json.loads(path.read_text())
        sample = path.parent.name
        rows = []
        with (path.parent / "drift.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["tensor"] == "hidden" and row["group"] in GROUPS and row["reference"] in {"previous", "initial"}:
                    rows.append(row)
        if not rows:
            continue
        layers = sorted({r["layer"] for r in rows})
        count = max(r["step"] for r in rows) + 1
        matrices = {(ref, group): np.full((len(layers), count), np.nan)
                    for ref in ("previous", "initial") for group in GROUPS}
        for row in rows:
            matrices[row["reference"], row["group"]][layers.index(row["layer"]), row["step"]] = \
                row["metrics"]["cosine_distance"]["mean"]
        end = result.get("baseline_content_revealed_step")
        sample_report = {"id": result["id"], "layers": layers, "baseline_content_revealed_step": end,
                         "score": result["score"], "active_mean_cosine_distance": {}}
        for scope, last in (("full", count - 1), ("active", end)):
            if last is None:
                continue
            last = min(last, count - 1)
            if last < 1:
                continue
            vmax = max(float(np.nanmax(matrices["previous", g][:, 1:last + 1])) for g in GROUPS) if last else 1e-8
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
            for ax, group in zip(axes, GROUPS):
                image = ax.imshow(matrices["previous", group][:, 1:last + 1], aspect="auto",
                                  origin="lower", interpolation="nearest", vmin=0, vmax=max(vmax, 1e-8),
                                  extent=(.5, last + .5, -.5, len(layers) - .5))
                ax.set_yticks(range(len(layers)), layers)
                ax.set(title=group, xlabel="Baseline forward", ylabel="Measured transformer layer")
                if end is not None and scope == "full":
                    ax.axvline(end + .5, color="white", ls="--", lw=1)
            fig.colorbar(image, ax=list(axes), label="Mean token drift: 1 − cosine similarity")
            fig.suptitle(f"{result['id']} / {scope}: consecutive forwards; dashed line = content completion")
            fig.savefig(output / f"{sample}_layer_drift_{scope}.png", dpi=160)
            plt.close(fig)
        sample_report["depth_summary"] = depth_summary(matrices, layers, end)
        nonzero = [i for i, layer in enumerate(layers) if layer > 0] or list(range(len(layers)))
        sample_report["legacy_active_mean_layers"] = [layers[i] for i in nonzero]
        fig, ax = plt.subplots(figsize=(9, 4))
        for group in GROUPS:
            initial = 1 - np.nanmean(matrices["initial", group][nonzero], axis=0)
            ax.plot(range(count), initial, label=group)
            if end is not None and end > 0:
                active = matrices["previous", group][nonzero, 1:end + 1]
                sample_report["active_mean_cosine_distance"][group] = float(np.nanmean(active))
        if end is not None:
            ax.axvspan(end, count - 1, color="gray", alpha=.12, label="After baseline content completion")
        ax.set(xlabel="Baseline forward", ylabel="Mean token cosine similarity to forward 0",
               title=f"{result['id']}: initial-reference slice; equal weighting of nonzero measured layers")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(output / f"{sample}_initial_similarity.png", dpi=160); plt.close(fig)
        pair_path = path.parent / "pairwise_cosine.json"
        sample_report["pairwise_available"] = pair_path.exists()
        if pair_path.exists():
            data = json.loads(pair_path.read_text())
            steps = data["steps"]
            pooled = {g: np.nanmean([np.array(r["cosine_similarity"], dtype=float) for r in data["records"]
                                    if r["group"] == g], axis=0) for g in GROUPS}
            sample_report["pairwise_layers"] = data["layers"]
            sample_report["pairwise_steps"] = steps
            sample_report["pairwise_complete"] = data["complete"]
            for scope, indices in (("full", list(range(len(steps)))),
                                   ("active", [i for i, step in enumerate(steps) if end is not None and step <= end])):
                if not indices:
                    continue
                subset = {g: a[np.ix_(indices, indices)] for g, a in pooled.items()}
                for scale, vmin in (("shared", min(float(np.nanmin(a)) for a in subset.values())),
                                    ("paper_scale", .8)):
                    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
                    for ax, group in zip(axes, GROUPS):
                        image = ax.imshow(subset[group], origin="upper", interpolation="nearest",
                                          vmin=min(vmin, .999999), vmax=1.)
                        ticks = np.unique(np.linspace(0, len(indices) - 1, min(6, len(indices))).round().astype(int))
                        ax.set_xticks(ticks, [steps[indices[i]] for i in ticks])
                        ax.set_yticks(ticks, [steps[indices[i]] for i in ticks])
                        ax.set(title=group, xlabel="Baseline forward", ylabel="Baseline forward")
                    fig.colorbar(image, ax=list(axes), label="Mean token cosine similarity")
                    fig.suptitle(f"{result['id']} / {scope}: mean token cosine, equal layer weights {data['layers']}")
                    fig.savefig(output / f"{sample}_pairwise_{scope}_{scale}.png", dpi=160)
                    plt.close(fig)
            sample_report["pairwise_matrices_mean_across_layers"] = {
                g: [[float(v) if np.isfinite(v) else None for v in row] for row in a] for g, a in pooled.items()}
        summary["samples"].append(sample_report)
    depth = [s["depth_summary"] for s in summary["samples"] if s.get("depth_summary") is not None]
    if depth:
        layer_sets = {tuple(d["layers"]) for d in depth}
        if len(layer_sets) != 1:
            raise ValueError("Mixed layer coverage across samples; refusing an aggregate depth average")
        summary["aggregate_depth_summary"] = {
            "samples": len(depth), "layers": depth[0]["layers"],
            "all_32_layers_observed": all(d["all_32_layers_observed"] for d in depth),
            "pooling": "Equal example weights after per-example active-step and layer averages; includes layer 0",
            "active_mean_all_measured_layers": {
                g: float(np.mean([d["active_mean_all_measured_layers"][g] for d in depth])) for g in GROUPS},
            "active_mean_by_layer": {
                g: {str(layer): float(np.mean([d["active_mean_by_layer"][g][str(layer)] for d in depth]))
                    for layer in depth[0]["layers"]} for g in GROUPS}}
        fig, ax = plt.subplots(figsize=(9, 4))
        for g in GROUPS:
            ax.plot(depth[0]["layers"], [summary["aggregate_depth_summary"]["active_mean_by_layer"][g][str(l)]
                                      for l in depth[0]["layers"]], marker=".", label=g)
        ax.set(xlabel="Transformer block output (0 based)", ylabel="Mean consecutive-step cosine distance",
               title=f"Baseline drift by layer; equal weights across {len(depth)} examples")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(output / "aggregate_drift_by_layer.png", dpi=160); plt.close(fig)
    (output / "measurements.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Wrote cosine figures for {len(summary['samples'])} samples to {output}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    analyze(args.run, args.out)
