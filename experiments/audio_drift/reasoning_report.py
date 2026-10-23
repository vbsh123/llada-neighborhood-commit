"""Paired final-answer accuracy and measured cache latency for reasoning pilots."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def summarize_rows(paired):
    n = len(paired)
    result = {
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
        "same_forward_count_samples": sum(r["same_forward_count"] for r in paired),
        "strict_accuracy": sum(r["strict_correct"] for r in paired) / n,
        "format_normalized_samples": sum(r["format_normalized"] for r in paired)}
    replay = [r for r in paired if r["baseline_replay_seconds"] is not None and r["policy_replay_seconds"] is not None]
    if replay:
        result["fixed_input_replay"] = {
            "samples": len(replay), "forwards_total": sum(r["replay_forwards"] for r in replay),
            "forward_speedup_ratio_of_sums": sum(r["baseline_replay_seconds"] for r in replay) /
                                            sum(r["policy_replay_seconds"] for r in replay)}
    correct = [r for r in paired if r['baseline_correct']]
    if correct:
        result['baseline_correct_subset'] = {
            'samples': len(correct), 'accuracy': sum(r['correct'] for r in correct) / len(correct),
            'decode_speedup_ratio_of_sums': sum(r['baseline_seconds'] for r in correct) / sum(r['seconds'] for r in correct)}
    return result


def compare_depth(rows):
    """Pair each depth-aware run with its flat counterpart on identical IDs."""
    output = {}
    for cohort in ['all', *sorted({r['cohort'] for r in rows})]:
        selected = [r for r in rows if cohort == 'all' or r['cohort'] == cohort]
        lookup = {(r['id'], r['condition']): r for r in selected}
        comparisons = defaultdict(list)
        for r in selected:
            if r['flat_condition']:
                flat = lookup.get((r['id'], r['flat_condition']))
                if flat is not None:
                    comparisons[r['condition']].append((flat, r))
        output[cohort] = {}
        for name, pairs in comparisons.items():
            item = {'flat_condition': pairs[0][1]['flat_condition'], 'samples': len(pairs),
                    'flat_correct': sum(f['correct'] for f,d in pairs),
                    'depth_correct': sum(d['correct'] for f,d in pairs),
                    'flat_correct_to_depth_wrong': sum(f['correct'] and not d['correct'] for f,d in pairs),
                    'flat_wrong_to_depth_correct': sum(not f['correct'] and d['correct'] for f,d in pairs),
                    'depth_speedup_over_flat': sum(f['seconds'] for f,d in pairs)/sum(d['seconds'] for f,d in pairs)}
            replay = [(f,d) for f,d in pairs if f['policy_replay_seconds'] is not None and d['policy_replay_seconds'] is not None]
            if replay:
                item['replay_samples'] = len(replay)
                item['replay_depth_speedup_over_flat'] = sum(f['policy_replay_seconds'] for f,d in replay)/sum(d['policy_replay_seconds'] for f,d in replay)
            output[cohort][name] = item
    return output


def summarize(results):
    rows, by_policy = [], defaultdict(list)
    for result in results:
        if (result.get("score") or {}).get("metric") != "numeric":
            continue
        bench = result.get("benchmark", {})
        if "baseline_median_seconds" not in bench:
            continue
        replay = result.get("fixed_input_replay", {})
        replay_policies = {p["condition"]: p for p in replay.get("policies", [])}
        for policy in bench["policies"]:
            fixed = replay_policies.get(policy["condition"], {})
            row = {"id": result["id"], "condition": policy["condition"],
                   "cohort": result.get("cohort", "unspecified"),
                   "flat_condition": policy.get("flat_condition"),
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
                   "same_forward_count": policy["same_forward_count_as_baseline"],
                   "baseline_visible_answer_tokens": bench.get("baseline_visible_answer_tokens"),
                   "policy_visible_answer_tokens": policy.get("visible_answer_tokens"),
                   "baseline_strict_correct": result["score"].get("strict_correct", result["score"]["correct"]),
                   "strict_correct": policy["score"].get("strict_correct", policy["score"]["correct"]),
                   "format_normalized": policy["score"].get("format_normalized", False),
                   "replay_forwards": replay.get("forwards") if fixed else None,
                   "baseline_replay_seconds": replay.get("baseline_median_seconds") if fixed else None,
                   "policy_replay_seconds": fixed.get("median_seconds")}
            rows.append(row)
            by_policy[policy["condition"]].append(row)
    summary = {}
    for condition, paired in by_policy.items():
        summary[condition] = summarize_rows(paired)
        summary[condition]['cohorts'] = {
            cohort: summarize_rows([r for r in paired if r['cohort'] == cohort])
            for cohort in sorted({r['cohort'] for r in paired})}
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
    lines = ["# Spoken math decoding evaluation", "",
             "Final numerical answer accuracy normalizes supported literal labels and an optional comma after an explicit final marker. Strict scores remain in per-sample JSON/CSV. Parse failures count as incorrect. Paired baselines use the same examples. Speedup is summed paired baseline median decode time divided by summed policy median decode time. EOS stopping may produce different forward counts; inspect reasoning_pairs.csv. No-EOS flags identify possible output-budget exhaustion.", "",
             "| Policy | N | Baseline correct | Policy correct | Correct → wrong | Wrong → correct | Parse failures | Decode speedup |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        n = s['samples']
        lines.append(f"| {name} | {n} | {round(s['paired_baseline_accuracy']*n)} | {round(s['accuracy']*n)} | {s['correct_to_wrong']} | {s['wrong_to_correct']} | {s['parse_failures']} | {s['decode_speedup_ratio_of_sums']:.3f}× |")
    for cohort in sorted({r['cohort'] for r in rows} - {'unspecified'}):
        lines += ["", f"## Cohort: {cohort}", "",
                  "| Policy | N | Baseline correct | Policy correct | Decode speedup |",
                  "|---|---:|---:|---:|---:|"]
        for name, values in summary.items():
            c = values['cohorts'].get(cohort)
            if c:
                n = c['samples']
                lines.append(f"| {name} | {n} | {round(c['paired_baseline_accuracy']*n)} | {round(c['accuracy']*n)} | {c['decode_speedup_ratio_of_sums']:.3f}× |")
    comparisons = compare_depth(rows)
    (out / 'depth_comparisons.json').write_text(json.dumps(comparisons, indent=2) + "\n")
    for cohort, values in comparisons.items():
        if not values:
            continue
        lines += ["", f"## Flat versus depth-aware: {cohort}", "",
                  "| Depth variant | N | Flat correct | Depth correct | Depth speedup over flat |",
                  "|---|---:|---:|---:|---:|"]
        for name, c in values.items():
            lines.append(f"| {name} | {c['samples']} | {c['flat_correct']} | {c['depth_correct']} | {c['depth_speedup_over_flat']:.3f}× |")
    replay = {name: s["fixed_input_replay"] for name, s in summary.items() if "fixed_input_replay" in s}
    if replay:
        lines += ["", "## Equal-input forward replay", "",
                  "All policies receive identical baseline token states and forward counts. Timings exclude token commitment and do not grade replay outputs. This isolates forward-path savings from changes in generated answer length; it is separate from autonomous decode speedup above.", "",
                  "| Policy | Samples | Forwards total | Forward replay speedup |",
                  "|---|---:|---:|---:|"]
        for name, r in replay.items():
            lines.append(f"| {name} | {r['samples']} | {r['forwards_total']} | {r['forward_speedup_ratio_of_sums']:.3f}× |")
    lines += ["", "Development and fresh cohorts are reported separately when available. Policy selection on either cohort requires further independent validation. Baseline failures stay in primary results; baseline-correct subsets are secondary diagnostics in JSON."]
    (out / "reasoning_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
