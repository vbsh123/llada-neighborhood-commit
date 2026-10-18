"""Paired final-answer accuracy and measured cache latency for reasoning pilots."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def summarize(results):
    rows, by_policy = [], defaultdict(list)
    for result in results:
        if result.get("score", {}).get("metric") != "numeric":
            continue
        bench = result.get("benchmark", {})
        if "baseline_median_seconds" not in bench:
            continue
        for policy in bench["policies"]:
            row = {"id": result["id"], "condition": policy["condition"],
                   "target": result["target"], "baseline_prediction": result["score"]["prediction"],
                   "prediction": policy["score"]["prediction"],
                   "baseline_correct": result["score"]["correct"], "correct": policy["score"]["correct"],
                   "parse_failed": policy["score"]["parse_failed"],
                   "baseline_parse_failed": result["score"]["parse_failed"],
                   "baseline_no_eos": result["answer_has_no_eos"],
                   "policy_no_eos": policy.get("answer_has_no_eos"),
                   "baseline_seconds": bench["baseline_median_seconds"],
                   "seconds": policy["median_seconds"],
                   "baseline_forwards": bench["baseline_forwards"],
                   "policy_forwards": policy["accounting"][0]["forwards"],
                   "same_forward_count": policy["same_forward_count_as_baseline"]}
            rows.append(row)
            by_policy[policy["condition"]].append(row)
    summary = {}
    for condition, paired in by_policy.items():
        n = len(paired)
        summary[condition] = {
            "samples": n,
            "paired_baseline_accuracy": sum(r["baseline_correct"] for r in paired) / n,
            "accuracy": sum(r["correct"] for r in paired) / n,
            "correct_to_wrong": sum(r["baseline_correct"] and not r["correct"] for r in paired),
            "wrong_to_correct": sum(not r["baseline_correct"] and r["correct"] for r in paired),
            "baseline_parse_failures": sum(r["baseline_parse_failed"] for r in paired),
            "parse_failures": sum(r["parse_failed"] for r in paired),
            "baseline_no_eos": sum(r["baseline_no_eos"] for r in paired),
            "policy_no_eos": sum(r["policy_no_eos"] is True for r in paired),
            "decode_speedup_ratio_of_sums": sum(r["baseline_seconds"] for r in paired) / sum(r["seconds"] for r in paired),
            "same_forward_count_samples": sum(r["same_forward_count"] for r in paired)}
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    results = [json.loads(f.read_text()) for f in sorted((args.run / "samples").glob("*/result.json"))
               if not f.parent.name.startswith(".")]
    rows, summary = summarize(results)
    if not rows:
        raise ValueError("No completed numerical-answer cache benchmarks")
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    (out / "reasoning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "reasoning_pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Spoken math cache pilot", "",
             "Final numerical answer accuracy. Parse failures count as incorrect. Paired baselines use the same examples. Speedup is summed paired baseline median decode time divided by summed policy median decode time. EOS stopping may produce different forward counts; inspect reasoning_pairs.csv. No-EOS flags identify possible output-budget exhaustion.", "",
             "| Policy | N | Baseline correct | Cache correct | Correct → wrong | Wrong → correct | Parse failures | Decode speedup |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        n = s['samples']
        lines.append(f"| {name} | {n} | {round(s['paired_baseline_accuracy']*n)} | {round(s['accuracy']*n)} | {s['correct_to_wrong']} | {s['wrong_to_correct']} | {s['parse_failures']} | {s['decode_speedup_ratio_of_sums']:.3f}× |")
    lines += ["", "This small test-set pilot is exploratory. Choose promising schedules, then evaluate on fresh examples; do not report pilot tuning results as held-out validation."]
    (out / "reasoning_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
