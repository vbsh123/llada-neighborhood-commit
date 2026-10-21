"""Separate synchronized per-forward diagnostics and optional operator traces.

Continuous equal-input replay remains the primary forward-path speed measure.
Per-step synchronization deliberately changes dispatch/pipeline behavior.
"""
from contextlib import nullcontext
from collections import defaultdict
import statistics
import time

import torch

from .benchmark import policy_specs
from .budget import flop_estimate, row_budget
from .compact import PrefixCache
from .decode import forward_logits
from .profile import decoder_scopes


def phase_name(step, specification):
    if specification is None:
        return 'uncached'
    if step == 0:
        return 'initialization'
    refreshed = [s['group'] for s in specification['schedule']
                 if s['refresh_every'] and step % s['refresh_every'] == 0]
    return '+'.join(refreshed) if refreshed else 'cache_hit'


def phase_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['phase']].append(row)
    return {k: {'observations': len(v), 'wall_median_ms': statistics.median(r['wall_ms'] for r in v),
                'cuda_median_ms': statistics.median(r['cuda_ms'] for r in v)} for k, v in groups.items()}


@torch.no_grad()
def profile_refresh(llm, prepared, states, active_forwards, args, work, sample_id):
    from .run import write_json
    count = min(active_forwards, len(states), args.refresh_profile_steps)
    inputs = [x.to(prepared.prompt.device) for x in states[:count]]
    all_specs = policy_specs(args, sample_id)
    specs = [None, *[s for s in all_specs if not s.get('pruning')]]
    report = {'pruning_policies_excluded': [s['condition'] for s in all_specs if s.get('pruning')],
              'forwards': count, 'active_baseline_forwards': active_forwards,
              'timed_repeats': args.refresh_profile_repeats, 'conditions': [],
              'note': 'Separate diagnostic: each forward uses CUDA events and synchronized wall timing. '
                      'Do NOT substitute these timings for continuous replay or autonomous decode. '
                      'Inputs and forward counts are identical across conditions; outputs are not graded. '
                      'Cache setup/teardown are outside per-step timing; initialization forward is included. '
                      'FLOP ratios are arithmetic surrogates, not maximum achievable wall speedups.'}
    for spec in specs:
        name = spec['condition'] if spec else 'uncached'
        print(f'  refresh profile {name}: warmup + {args.refresh_profile_repeats} synchronized repeats; '
              f'{count} forwards each', flush=True)
        # Warm this complete pattern with a fresh cache, outside all timings.
        with PrefixCache(llm, prepared, spec['schedule'], pruning=spec.get('pruning')) if spec else nullcontext():
            for x in inputs:
                output = forward_logits(llm, prepared, x)
                del output
        records = []
        for repeat in range(args.refresh_profile_repeats):
            with (PrefixCache(llm, prepared, spec['schedule'], pruning=spec.get('pruning')) if spec else nullcontext()) as cache:
                for step, x in enumerate(inputs):
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    start_event.record()
                    output = forward_logits(llm, prepared, x)
                    end_event.record()
                    end_event.synchronize()
                    elapsed = (time.perf_counter() - began) * 1000
                    record = {'repeat': repeat, 'step': step, 'phase': phase_name(step, spec),
                              'wall_ms': elapsed, 'cuda_ms': start_event.elapsed_time(end_event)}
                    if spec:
                        event = cache.events[-1]
                        record.update(computed_rows_by_layer=event['layer_computed_rows'],
                                      refreshed_by_layer=event['layer_refresh'])
                    records.append(record)
                    del output
                if spec:
                    expected = row_budget(prepared, args.length, count, spec, llm.config.n_layers)
                    if expected['computed_rows_by_layer'] != cache.accounting()['computed_rows_by_layer']:
                        raise RuntimeError('Analytical budget disagrees with actual compact row execution')
            print(f'    refresh profile repeat {repeat + 1}/{args.refresh_profile_repeats} complete', flush=True)
        budget = row_budget(prepared, args.length, count, spec or {'schedule': []}, llm.config.n_layers)
        item = {'condition': name, 'records': records, 'phases': phase_summary(records),
                'row_budget': budget, 'flop_estimate': flop_estimate(llm, budget)}
        report['conditions'].append(item)
        write_json(work / 'refresh_profile.json', report)
        print(f'  refresh timing {name}: {count} states; phases={list(item["phases"])}', flush=True)
    if args.refresh_trace:
        capture_traces(llm, prepared, inputs, specs, work)
    return report


@torch.no_grad()
def capture_traces(llm, prepared, inputs, specs, work):
    """Retain cache history, but profile ONLY representative forwards, not setup."""
    chosen = {0, 1, 8, 16, 32}
    keep = {'freeze_prefix_r8', 'freeze_triple_a16_p8_s16', 'freeze_triple_a32_p8_s32'}
    directory = work / 'refresh_traces'
    directory.mkdir(exist_ok=True)
    index = []
    for spec in specs:
        name = spec['condition'] if spec else 'uncached'
        if spec and name not in keep:
            continue
        with PrefixCache(llm, prepared, spec['schedule'], pruning=spec.get('pruning')) if spec else nullcontext():
            for step, x in enumerate(inputs[:max(chosen) + 1]):
                if step not in chosen:
                    output = forward_logits(llm, prepared, x)
                    del output
                    continue
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                    with decoder_scopes(llm):
                        with torch.profiler.record_function('diagnostic_forward'):
                            output = forward_logits(llm, prepared, x)
                            del output
                    torch.cuda.synchronize()
                filename = f'{name}_step{step:03d}'
                profiler.export_chrome_trace(str(directory / (filename + '.json')))
                events = profiler.key_averages()
                (directory / (filename + '.txt')).write_text(events.table(sort_by='self_cuda_time_total', row_limit=50))
                index.append({'condition': name, 'step': step, 'phase': phase_name(step, spec),
                              'trace': filename + '.json',
                              'operators': [{'name': e.key, 'calls': e.count,
                                  'self_device_us': e.self_device_time_total} for e in events],
                              'scopes': [{'name': e.key, 'calls': e.count, 'device_total_us': e.device_time_total}
                                         for e in events if e.key.startswith('decoder.')]})
    from .run import write_json
    write_json(directory / 'index.json', {'captures': index,
        'note': 'Instrumented timings are not latency benchmarks. Scope totals overlap with their operators; '
                'never add both. Sum disjoint kernel durations from Chrome traces for GPU work, not enclosing spans.'})
