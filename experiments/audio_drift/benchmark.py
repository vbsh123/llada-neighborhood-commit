"""Warmed batch-one decoding timings with actual fixed-prefix row skipping."""
from contextlib import nullcontext
import hashlib
import random
import statistics
import time

import torch

from .compact import policy_context
from .decode import decode
from .inputs import score
from .protocol import is_identity


def visible_ids(final, prepared, eos_id):
    ids = final[0, prepared.prompt.shape[1]:].tolist()
    return ids[:ids.index(eos_id) + 1] if eos_id in ids else ids


def policy_specs(args, sample_id=None):
    from .run import intervention_specs
    specs = [s for s in intervention_specs(args) if s["mode"] == "freeze"]
    names = getattr(args, "benchmark_conditions", [])
    if names:
        unknown = set(names) - {s["condition"] for s in specs}
        if unknown:
            raise ValueError(f"Unknown benchmark conditions: {sorted(unknown)}")
        specs = [s for s in specs if s["condition"] in names]
    if not specs:
        raise ValueError("Benchmark needs at least one freeze policy")
    if sample_id is not None and any(s.get('pruning') for s in specs):
        # Paired scope variants share a sample-specific seed; repeats are exact.
        prune_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{sample_id}:pruning".encode()).digest()[:8], 'big') % (2**63)
        specs = [{**s, 'pruning': {**s['pruning'], 'seed': prune_seed}} if s.get('pruning') else s for s in specs]
    if getattr(args, 'shuffle_policy_order', False):
        if sample_id is None:
            raise ValueError("Shuffled timing order requires a sample ID")
        seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{sample_id}".encode()).digest()[:8], 'big')
        identities = [s for s in specs if is_identity(s)]
        rest = [s for s in specs if not is_identity(s)]
        random.Random(seed).shuffle(rest)
        specs = identities + rest
    from .crossing_pruning import dependency_order
    return dependency_order(specs)


@torch.no_grad()
def benchmark(llm, prepared, official_baseline, args, tokenizer, row, work, diagnostics=(), preparation_seconds=None):
    from .run import display_text, write_json
    specs = policy_specs(args, row.get('id'))
    eos = tokenizer.eos_token_id if args.benchmark_stopping == "eos" else None

    def measure(spec=None):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        context = policy_context(llm, prepared, spec) if spec else nullcontext()
        with context as cache:
            final, _, events = decode(llm, prepared, args.length, args.block_length, args.steps,
                                     save_states=False, eos_id=eos)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        counts = cache.accounting() if spec else {"forwards": len(events)}
        if spec and spec.get('pruning'):
            for pick in counts['pruning']['selections']:
                anchor = events[pick['step']]
                pick['anchor_progress'] = {key: anchor[key] for key in
                    ('remaining_masks', 'remaining_fraction', 'active_remaining_masks', 'active_remaining_fraction',
                     'block', 'step_in_block', 'previous_changed_tokens')}
                pick['anchor_progress']['revealed_fraction'] = 1 - anchor['remaining_fraction']
        return final, elapsed, peak, counts

    print(f"  benchmark: {len(specs)} policies; {args.benchmark_repeats} timed repeats; "
          f"stopping={args.benchmark_stopping}; no drift hooks", flush=True)
    warm, _, _, _ = measure()
    expected = visible_ids(official_baseline, prepared, tokenizer.eos_token_id)
    if visible_ids(warm, prepared, tokenizer.eos_token_id) != expected:
        raise RuntimeError("Benchmark stopping changed baseline display tokens")
    del warm
    # Bracket the policy measurements with warmed baseline timings to expose
    # gross GPU clock drift. Report individual repeats, never select the fastest.
    baseline_before, before, _, base_count = measure()
    del baseline_before
    report = {"policy_order": [s["condition"] for s in specs],
              "stopping": args.benchmark_stopping, "timed_repeats_per_policy": args.benchmark_repeats,
              "shared_audio_preparation_seconds": preparation_seconds,
              "baseline_seconds": [before], "baseline_forwards": base_count["forwards"],
              "baseline_visible_answer_tokens": len(expected) - int(tokenizer.eos_token_id in expected),
              "policies": [], "note": "Warmed, synchronized batch-one wall time including cache initialization, "
              "refreshes, gathers/scatters, ordinary decoder bookkeeping and full vocabulary head. "
              "Model loading, disk I/O, audio preparation, drift collection and task scoring are excluded. "
              "EOS stopping waits until all positions before a revealed EOS are filled; baseline and "
              "cache use the same rule. Compare latency together with task quality and forwards. Policies can be "
              "slower. The skipped-token-update fraction is not a wall-time prediction."}
    write_json(work / "cache_benchmark.json", report)
    for spec in specs:
        from .crossing_pruning import resolve_control
        spec = resolve_control(spec, report['policies'])
        print(f"    benchmark {spec['condition']}: warmup starting",flush=True)
        warm, _, _, warm_counts = measure(spec)
        del warm
        if spec.get('pruning',{}).get('progressive_global'):
            picks=warm_counts['pruning']['selections']
            print(f"      warmup: {sum(p['drop_budget'] for p in picks)} prefix positions removed, "
                  f"{len(picks)} applied batches; skipped row updates={warm_counts['skipped_token_update_fraction']:.2%}",flush=True)
        durations, outputs, accounting, peaks = [], [], [], []
        for _ in range(args.benchmark_repeats):
            final, elapsed, peak, counts = measure(spec)
            durations.append(elapsed)
            outputs.append(visible_ids(final, prepared, tokenizer.eos_token_id))
            accounting.append(counts)
            peaks.append(peak)
            print(f"      timed repeat {len(durations)}/{args.benchmark_repeats}: {elapsed:.2f}s; {counts['forwards']} forwards",flush=True)
            text = display_text(tokenizer, final, prepared.prompt.shape[1])
            del final
        reproducible = all(x == outputs[0] for x in outputs)
        if spec.get('pruning') and any(a['pruning'] != accounting[0]['pruning'] for a in accounting):
            raise RuntimeError(f"Timed repeats changed pruning selections: {spec['condition']}")
        if not reproducible:
            raise RuntimeError(f"Timed repeats changed answer tokens: {spec['condition']}")
        if is_identity(spec) and outputs[0] != expected:
            raise RuntimeError("Optimized refresh-every-forward identity differs from baseline")
        result = {**spec, "seconds": durations, "median_seconds": statistics.median(durations),
                  "peak_allocated_bytes": max(peaks), "accounting": accounting,
                  "text": text, "visible_answer_ids": outputs[0], "same_visible_tokens": outputs[0] == expected,
                  "score": score(text, row.get("target"), row.get("metric", "none")),
                  "answer_has_no_eos": tokenizer.eos_token_id not in outputs[0],
                  "visible_answer_tokens": len(outputs[0]) - int(tokenizer.eos_token_id in outputs[0]),
                  "selected_positions": {s["group"]: prepared.group_positions(s["group"]) for s in spec["schedule"]}}
        paired = next((b for b in diagnostics if b["condition"] == spec["condition"] and b["anchor_step"] == 0), None)
        if paired:
            dense_ids = paired["answer_ids"]
            if tokenizer.eos_token_id in dense_ids:
                dense_ids = dense_ids[:dense_ids.index(tokenizer.eos_token_id) + 1]
            result["same_visible_tokens_as_dense_intervention"] = outputs[0] == dense_ids
        report["policies"].append(result)
        write_json(work / "cache_benchmark.json", report)
        print(f"    benchmark {spec['condition']}: {result['median_seconds']:.2f}s; "
              f"{accounting[0]['forwards']} forwards; score={result['score']}", flush=True)
    baseline_after, after, _, after_count = measure()
    if visible_ids(baseline_after, prepared, tokenizer.eos_token_id) != expected:
        raise RuntimeError("Post-benchmark baseline changed")
    del baseline_after
    report["baseline_seconds"].append(after)
    report["baseline_after_forwards"] = after_count["forwards"]
    baseline_seconds = statistics.median(report["baseline_seconds"])
    report["baseline_median_seconds"] = baseline_seconds
    for result in report["policies"]:
        result["decode_speedup"] = baseline_seconds / result["median_seconds"]
        if preparation_seconds is not None:
            result["speedup_with_shared_preparation"] = ((baseline_seconds + preparation_seconds) /
                                                        (result["median_seconds"] + preparation_seconds))
        result["same_forward_count_as_baseline"] = all(a["forwards"] == base_count["forwards"] for a in result["accounting"])
    write_json(work / "cache_benchmark.json", report)
    return report
