"""Posthoc baseline stability times. Thresholds are diagnostics, not cache decisions."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def timing(series, threshold, window=4):
    """Require W consecutive observed steps; an isolated dip is not stability.

    'through_end' needs a qualifying window AND no later violations. This is
    a retrospective description; it cannot be used as an online oracle.
    """
    if window < 1 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid stability threshold or window")
    points = sorted(series)
    if len({s for s, _ in points}) != len(points) or any(not math.isfinite(v) for _, v in points):
        raise ValueError("Stability series needs unique steps and finite values")
    if not points:
        return {"observations": 0, "first_sustained_step": None,
                "stable_through_end_step": None, "fraction_under_threshold": None}
    run = []
    candidates = []
    last_violation = None
    for step, value in points:
        if value <= threshold:
            if run and step != run[-1] + 1:
                run = []
            run.append(step)
            if len(run) >= window:
                candidates.append(run[-window])
        else:
            last_violation = step
            run = []
    permanent = [s for s in candidates if last_violation is None or s > last_violation]
    return {"observations": len(points), "first_sustained_step": min(candidates) if candidates else None,
            "stable_through_end_step": min(permanent) if permanent else None,
            "last_violation_step": last_violation,
            "fraction_under_threshold": sum(v <= threshold for _, v in points) / len(points),
            "mean_drift": statistics.mean(v for _, v in points),
            "max_drift": max(v for _, v in points)}


def analyze(run, thresholds=(.01, .02, .05), window=4,
            metric="relative_l2", statistic="p90"):
    if metric not in {"relative_l2", "cosine_distance"} or statistic not in {"median", "p90", "max"}:
        raise ValueError("Unsupported metric or token statistic")
    records = []
    for path in sorted((run / "samples").glob("*/result.json")):
        if path.parent.name.startswith("."):
            continue
        result = json.loads(path.read_text())
        series = defaultdict(list)
        with (path.parent / "drift.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["age"] == 0 or row["group"] not in {"audio_acoustic", "audio_semantic", "prompt"}:
                    continue
                key = row["group"], row["tensor"], row["reference"], row["layer"]
                series[key].append((row["step"], row["metrics"][metric][statistic]))
        for (group, tensor, reference, layer), values in sorted(series.items()):
            for threshold in thresholds:
                records.append({"id": result["id"], "task": result["task"], "group": group,
                                "tensor": tensor, "reference": reference, "layer": layer,
                                "metric": metric, "token_statistic": statistic,
                                "threshold": threshold, "window": window,
                                **timing(values, threshold, window)})
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    destination = output / f"stability_{metric}_{statistic}.json"
    destination.write_text(json.dumps({"note": "Posthoc threshold sweep, not an adaptive policy or quality guarantee. "
                          "Age-zero comparisons are excluded; windows require consecutive steps. "
                          "A stable tail may coincide with already-completed answer content.",
                          "records": records}, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(records)} stability measurements to {destination}", flush=True)
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[.01, .02, .05])
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--metric", choices=["relative_l2", "cosine_distance"], default="relative_l2")
    p.add_argument("--statistic", choices=["median", "p90", "max"], default="p90")
    args = p.parse_args()
    analyze(args.run, args.thresholds, args.window, args.metric, args.statistic)
