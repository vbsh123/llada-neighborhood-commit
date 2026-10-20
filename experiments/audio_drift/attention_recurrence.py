"""Audio attention transitions from low to high and from high to low.

Offline baseline analysis, not a pruning policy or proof of causal irrelevance.
Uses the same fixed answer-probe population at every forward (schema version 2).
"""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import statistics

import numpy as np


def low_episode_anchors(low, high, history):
    """First eligible point per contiguous low run, for each token separately.

    A trigger must have history consecutive low means and must not already be
    high under the chosen future-high metric. Different history settings define
    separate analyses. Multiple episodes for one token are not independent.
    """
    low, high = np.asarray(low, dtype=bool), np.asarray(high, dtype=bool)
    if low.shape != high.shape or low.ndim != 2 or history < 1:
        raise ValueError('Matching [forwards,tokens] masks and positive history required')
    run = np.zeros(low.shape[1], dtype=np.int32)
    anchored = np.zeros(low.shape[1], dtype=bool)
    times, tokens = [], []
    for t in range(len(low)):
        run = np.where(low[t], run + 1, 0)
        anchored &= low[t]
        eligible = (run >= history) & ~high[t] & ~anchored
        selected = np.flatnonzero(eligible)
        times.extend([t] * len(selected)); tokens.extend(selected.tolist())
        anchored |= eligible
    return np.asarray(times, dtype=np.int32), np.asarray(tokens, dtype=np.int32)


def reactivation_records(mean, peak, history=4, low_threshold=.5, high_threshold=2.,
                         horizons=(4, 8, 16, 32), high_metric='mean', direction='low_to_high'):
    """Inputs are attention weights divided by uniform attention 1/Nkeys."""
    mean, peak = np.asarray(mean, dtype=float), np.asarray(peak, dtype=float)
    if (mean.shape != peak.shape or mean.ndim != 2 or not mean.size or
            not np.isfinite(mean).all() or not np.isfinite(peak).all() or
            np.any(mean < 0) or np.any(peak + 1e-6 * np.maximum(1., peak) < mean) or
            not 0 <= low_threshold < high_threshold or high_metric not in {'mean', 'peak'} or
            direction not in {'low_to_high', 'high_to_low'} or
            not horizons or any(h < 1 for h in horizons)):
        raise ValueError('Invalid attention series, metrics, thresholds or horizons')
    high_values = mean if high_metric == 'mean' else peak
    high = high_values >= high_threshold
    low = (mean if direction == 'low_to_high' else high_values) <= low_threshold
    source, target = (low, high) if direction == 'low_to_high' else (high, low)
    times, tokens = low_episode_anchors(source, target, history)
    n, m = mean.shape
    next_high = np.full((n, m), n, dtype=np.int32)
    last = np.full(m, n, dtype=np.int32)
    for t in range(n - 1, -1, -1):
        next_high[t] = last
        last = np.where(target[t], t, last)
    future = next_high[times, tokens]
    delays = future - times
    has_future_high = future < n
    quarters = np.minimum(3, (times * 4 // max(1, n)))
    rows = []
    for quarter in [-1, 0, 1, 2, 3]:
        chosen = np.ones(len(times), dtype=bool) if quarter == -1 else quarters == quarter
        for horizon in [*horizons, 'remaining_decode']:
            eligible = chosen & ((times + horizon < n) if isinstance(horizon, int) else (times + 1 < n))
            hit = eligible & has_future_high
            if isinstance(horizon, int):
                hit &= delays <= horizon
            selected_delays = delays[hit].tolist()
            count, hits = int(eligible.sum()), int(hit.sum())
            histogram = {str(k): v for k, v in sorted(Counter(selected_delays).items())}
            rows.append({'direction': direction, 'history': history, 'high_metric': high_metric, 'horizon': horizon,
                         'anchor_decode_quarter': quarter, 'low_threshold_uniform_units': low_threshold,
                         'high_threshold_uniform_units': high_threshold,
                         'source_episodes': int(chosen.sum()),
                         'low_episodes': int(chosen.sum()) if direction == 'low_to_high' else None,
                         'eligible_episodes': count,
                         'right_censored_episodes': int(chosen.sum()) - count,
                         'transitioned_episodes': hits,
                         'reactivated_episodes': hits if direction == 'low_to_high' else None,
                         'probability': hits / count if count else None,
                         'distinct_eligible_tokens': int(len(np.unique(tokens[eligible]))),
                         'first_target_delay_histogram': histogram,
                         'first_high_delay_histogram': histogram if direction == 'low_to_high' else {}})
    return rows


def analyze(run, out=None, histories=(1, 4, 8), horizons=(4, 8, 16, 32),
            low=.5, high=2., plots=True):
    from .diagnostic_report import Archive, write_csv
    if (not histories or not horizons or any(v < 1 for v in [*histories, *horizons]) or
            len(set(histories)) != len(histories) or len(set(horizons)) != len(horizons) or
            not 0 <= low < high):
        raise ValueError('Distinct positive histories/horizons and ordered nonnegative thresholds required')
    archive = Archive(run)
    if out is None and archive.zip:
        archive.close()
        raise ValueError('ZIP analysis requires --out')
    out = Path(out) if out else Path(run)/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    names = set(archive.names())
    rows, unsupported, sample_count = [], [], 0
    try:
        result_names = sorted(n for n in names if n.startswith('samples/') and n.endswith('/result.json')
                              and not Path(n).parts[1].startswith('.'))
        for name in result_names:
            result = archive.read(name); folder = name.rsplit('/', 1)[0]
            paths = sorted(n for n in names if n.startswith(folder + '/attention_token_metrics/') and n.endswith('.npz'))
            if not paths:
                continue
            series = defaultdict(list)
            cutoff = result.get('baseline_content_revealed_step')
            for path in paths:
                step = int(Path(path).stem)
                if cutoff is not None and step > cutoff:
                    continue
                with archive.stream(path) as f, np.load(io.BytesIO(f.read()), allow_pickle=False) as data:
                    if 'attention_schema_version' not in data or int(data['attention_schema_version']) < 2:
                        unsupported.append(result['id']); series.clear(); break
                    n = int(data['sequence_length'])
                    for key in data.files:
                        suffix = '_attention_all_probe_mean'
                        if not key.endswith(suffix):
                            continue
                        prefix = key[:-len(suffix)]
                        layer, group = prefix.split('_', 1)
                        if group not in {'acoustic', 'semantic'}:
                            continue
                        series[(int(layer[1:]), group)].append((step, data[prefix + '_positions'].copy(),
                            data[key].astype(float) * n, data[prefix + '_attention_all_probe_peak'].astype(float) * n))
            if not series:
                continue
            sample_count += 1
            for (layer, group), values in series.items():
                steps = [v[0] for v in values]
                if steps != list(range(len(values))) or any(not np.array_equal(v[1], values[0][1]) for v in values):
                    raise ValueError(f'Nonconsecutive attention observations or changed positions: {result["id"]}, {layer}, {group}')
                mean, peak = np.stack([v[2] for v in values]), np.stack([v[3] for v in values])
                for history in histories:
                    for metric in ['mean', 'peak']:
                        for direction in ['low_to_high', 'high_to_low']:
                            for record in reactivation_records(mean, peak, history, low, high, horizons, metric, direction):
                                rows.append({'id': result['id'], 'cohort': result.get('cohort'),
                                             'layer': layer, 'group': group, 'observed_forwards': len(values), **record})
        aggregate = []
        grouping = defaultdict(list)
        for row in rows:
            key = (row['direction'], row['group'], row['layer'], row['history'], row['high_metric'],
                   row['horizon'], row['anchor_decode_quarter'])
            grouping[key].append(row)
        for (direction, group, layer, history, metric, horizon, quarter), values in grouping.items():
            eligible = sum(v['eligible_episodes'] for v in values)
            hits = sum(v['transitioned_episodes'] for v in values)
            probabilities = [v['probability'] for v in values if v['probability'] is not None]
            aggregate.append({'direction': direction, 'group': group, 'layer': layer, 'history': history,
                              'high_metric': metric, 'horizon': horizon, 'anchor_decode_quarter': quarter,
                              'eligible_samples': len(probabilities), 'eligible_episodes': eligible,
                              'transitioned_episodes': hits,
                              'reactivated_episodes': hits if direction == 'low_to_high' else None,
                              'right_censored_episodes': sum(v['right_censored_episodes'] for v in values),
                              'pooled_probability': hits / eligible if eligible else None,
                              'equal_sample_mean_probability': statistics.mean(probabilities) if probabilities else None})
        notes = [
            'Low = mean attention across the SAME fixed probe queries and all heads <= low/Nkeys. '
            'Future high is either mean or maximum across those probes/heads >= high/Nkeys. '
            'For high-to-low, low uses that SAME metric, so peak-to-low requires the peak itself to fall.',
            'Thresholds are descriptive choices, not calibrated pruning guarantees. Rerun this CPU analysis with other thresholds.',
            'Both low-to-high and high-to-low directions are analyzed. One trigger per contiguous source-state '
            'episode per token/history/high metric; positions already in the target state are excluded.',
            'Reaching low attention once after a high episode does not mean it remains low; use low-to-high risk separately.',
            'A fixed horizon requires the entire future window. Near-end episodes are reported as censored, not counted as safe.',
            'remaining_decode covers only the observed remaining baseline decode; exposure lengths differ.',
            'Separate sample/layer/group results; multiple token episodes and layers are dependent. No independent-token confidence claim.',
            'Only sampled answer queries are covered; low attention here does not imply low attention from every answer position.',
            'Baseline attention can change after pruning. These observational risks require actual pruning intervention validation.',
            'Post-content-completion observations are excluded when the saved result supplies that cutoff.',
            'Schema-1 conditional-query means are unsupported: query selection changed across forwards. They require recollection, not relabeling.'
        ]
        write_csv(out/'attention_recurrence_per_sample.csv', [
            {**r, 'first_high_delay_histogram': json.dumps(r['first_high_delay_histogram']),
             'first_target_delay_histogram': json.dumps(r['first_target_delay_histogram'])} for r in rows])
        write_csv(out/'attention_recurrence_by_layer.csv', aggregate)
        report = {'samples': sample_count, 'unsupported_samples': sorted(set(unsupported)),
                  'low_threshold_uniform_units': low, 'high_threshold_uniform_units': high,
                  'histories': list(histories), 'horizons': list(horizons), 'notes': notes, 'records': aggregate}
        (out/'attention_recurrence_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        (out/'attention_recurrence_report.md').write_text('# Future attention transitions in both directions\n\n'
            +f'Analyzed examples: {sample_count}. Unsupported older examples: {len(set(unsupported))}.\n\n'
            +'\n\n'.join(notes)+'\n')
        if plots and aggregate:
            for direction in ['low_to_high', 'high_to_low']:
                plot(out, [r for r in aggregate if r['direction'] == direction], low, high, direction)
        print(f'Wrote audio attention recurrence analysis: {sample_count} examples, {len(aggregate)} layer conditions', flush=True)
        return report
    finally:
        archive.close()


def plot(out, records, low, high, direction='low_to_high'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    histories = {r['history'] for r in records}
    history = 4 if 4 in histories else min(histories)
    horizons = sorted({r['horizon'] for r in records if isinstance(r['horizon'], int)})
    for i, group in enumerate(['acoustic', 'semantic']):
        for j, metric in enumerate(['mean', 'peak']):
            ax = axes[i, j]
            for horizon in horizons:
                selected = sorted([r for r in records if r['group'] == group and r['high_metric'] == metric
                    and r['history'] == history and r['horizon'] == horizon and r['anchor_decode_quarter'] == -1
                    and r['equal_sample_mean_probability'] is not None], key=lambda r:r['layer'])
                ax.plot([r['layer'] for r in selected], [r['equal_sample_mean_probability'] for r in selected],
                        label=f'next {horizon} forwards')
            label = f'low mean → high {metric}' if direction == 'low_to_high' else f'high {metric} → low {metric}'
            ax.set_title(f'{group}: {label}')
            ax.set_xlabel('Transformer layer'); ax.set_ylabel('Probability of reaching target attention state')
            ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle(f'{history} consecutive source-state forwards; low ≤ {low}×uniform, high ≥ {high}×uniform\n'
                 'Equal sample means; full future windows only; sampled answer queries')
    filename = 'attention_recurrence.png' if direction == 'low_to_high' else 'attention_high_to_low.png'
    fig.tight_layout(); fig.savefig(out/filename, dpi=160); plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--out', type=Path)
    p.add_argument('--histories', type=int, nargs='+', default=[1,4,8])
    p.add_argument('--horizons', type=int, nargs='+', default=[4,8,16,32])
    p.add_argument('--low', type=float, default=.5); p.add_argument('--high', type=float, default=2.)
    p.add_argument('--no-plots', action='store_true')
    args=p.parse_args()
    analyze(args.run,args.out,args.histories,args.horizons,args.low,args.high,not args.no_plots)
