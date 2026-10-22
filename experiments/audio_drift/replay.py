"""Equal-input forward timing, independent of a cache policy's answer length.

Every policy receives the same baseline token states in the same order. It never
commits predicted tokens, grades answers, or feeds replay outputs into generation.
This measures forward-path savings, not autonomous end-to-end decoding latency.
"""
from contextlib import nullcontext
import statistics
import time

import torch

from .compact import policy_context
from .decode import forward_logits
from .protocol import is_identity


@torch.no_grad()
def benchmark_replay(llm, prepared, states, baseline_forwards, args, work, sample_id=None):
    from .run import write_json
    from .benchmark import policy_specs
    count = min(len(states), baseline_forwards, args.replay_steps)
    if count < 1:
        raise ValueError("Replay needs at least one active baseline input")
    inputs = [x.to(prepared.prompt.device) for x in states[:count]]
    specs = policy_specs(args, sample_id)

    def measure(spec=None):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        context = policy_context(llm, prepared, spec) if spec else nullcontext()
        with context as cache:
            last = None
            for x in inputs:
                # Release the previous vocabulary output before the next forward.
                del last
                last = forward_logits(llm, prepared, x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        peak = torch.cuda.max_memory_allocated()
        accounting = cache.accounting() if spec else {'forwards': count}
        return last, elapsed, peak, accounting

    print(f"  fixed-input replay: {count} identical baseline states; {len(specs)} policies; "
          f"{args.replay_repeats} timed repeats; excludes token commitment and task scoring", flush=True)
    warm, _, _, _ = measure()
    expected_last = warm.argmax(-1)
    del warm
    before, before_seconds, _, _ = measure()
    if not torch.equal(before.argmax(-1), expected_last):
        raise RuntimeError("Replay baseline warmup changed final-step predictions")
    del before
    report = {'policy_order': [s['condition'] for s in specs], 'forwards': count, 'baseline_active_forwards': baseline_forwards,
              'step_indices': list(range(count)), 'repeats': args.replay_repeats,
              'baseline_seconds': [before_seconds], 'policies': [],
              'protocol': 'same-baseline-inputs-v1',
              'note': 'Identical preloaded token states, sequence length, forward count and full vocabulary head. '
                      'Includes cache initialization and forward-path Python/gather/scatter overhead. '
                      'Excludes CPU/GPU input transfers, token commitment, drift hooks and task scoring. '
                      'Only the initial baseline pre-EOS region is replayed. This is a forward-path benchmark, '
                      'not a quality measurement or autonomous decode speedup.'}
    path = work / 'fixed_input_replay.json'
    write_json(path, report)
    for spec in specs:
        warm, _, _, _ = measure(spec)
        identity = is_identity(spec)
        identity_match = torch.equal(warm.argmax(-1), expected_last) if identity else None
        del warm
        if identity and not identity_match:
            raise RuntimeError("Replay refresh-every-forward identity changed final-step predictions")
        durations, peaks, accounting = [], [], []
        for _ in range(args.replay_repeats):
            last, seconds, peak, counts = measure(spec)
            del last
            if counts['forwards'] != count:
                raise RuntimeError("Replay cache forward count differs from baseline")
            durations.append(seconds); peaks.append(peak); accounting.append(counts)
        result = {**spec, 'seconds': durations, 'median_seconds': statistics.median(durations),
                  'peak_allocated_bytes': max(peaks), 'accounting': accounting,
                  'identity_last_step_top1_match': identity_match}
        report['policies'].append(result)
        write_json(path, report)
        print(f"    replay {spec['condition']}: {result['median_seconds']:.2f}s; {count} forwards", flush=True)
    peers = {p['condition']: p for p in report['policies']}
    for policy in report['policies']:
        paired = peers.get(policy.get('paired_condition'))
        if paired:
            for a,b in zip(policy['accounting'],paired['accounting']):
                if (a['computed_rows_by_layer'] != b['computed_rows_by_layer'] or
                        a['qk_query_key_pairs_by_layer'] != b['qk_query_key_pairs_by_layer']):
                    raise RuntimeError('Staged generic replay control has a different execution budget')
    after, after_seconds, _, _ = measure()
    if not torch.equal(after.argmax(-1), expected_last):
        raise RuntimeError("Replay post-policy baseline changed final-step predictions")
    del after
    report['baseline_seconds'].append(after_seconds)
    report['baseline_median_seconds'] = statistics.median(report['baseline_seconds'])
    for result in report['policies']:
        result['forward_replay_speedup'] = report['baseline_median_seconds'] / result['median_seconds']
    write_json(path, report)
    return report
