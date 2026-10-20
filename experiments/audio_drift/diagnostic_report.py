"""CPU analysis of token/refresh budgets, drift/progress and attention dynamics.

Accepts completed directories or ZIP exports; streams old drift JSONL directly
from ZIP rather than extracting multi-gigabyte representation statistics.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import zipfile

from .budget import row_budget


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None
        if self.zip:
            names = [n for n in self.zip.namelist() if n.endswith('/config.json') or n == 'config.json']
            if len(names) != 1:
                raise ValueError('Expected exactly one root config.json in archive')
            self.prefix = names[0][:-len('config.json')]
        else:
            self.prefix = ''

    def names(self):
        if self.zip:
            return [n[len(self.prefix):] for n in self.zip.namelist() if n.startswith(self.prefix)]
        return [str(p.relative_to(self.path)) for p in self.path.rglob('*') if p.is_file()]

    @contextmanager
    def stream(self, name):
        with self.zip.open(self.prefix + name) if self.zip else (self.path / name).open('rb') as f:
            yield f

    def read(self, name):
        with self.stream(name) as f:
            return json.load(f)

    def close(self):
        if self.zip:
            self.zip.close()


def summary(values):
    return {'count': len(values), 'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values)} if values else None


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def analyze(run, out=None, plots=True):
    from .attention_dynamics import correlations
    archive = Archive(run)
    if out is None and archive.zip:
        archive.close()
        raise ValueError('ZIP analysis requires --out')
    out = Path(out) if out else Path(run) / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    names = set(archive.names())
    results = [archive.read(n) for n in sorted(names) if n.startswith('samples/') and n.endswith('/result.json')
               and not Path(n).parts[1].startswith('.')]
    if not results:
        archive.close()
        raise ValueError('No completed samples')
    budget_rows, dynamics, attention_rows, phase_rows, cost_rows, headroom_rows = [], [], [], [], [], []
    pairs, progress_bins, temporal_series = defaultdict(list), defaultdict(list), defaultdict(list)
    files = {archive.read(n)['id']: n.rsplit('/', 1)[0] for n in sorted(names)
             if n.startswith('samples/') and n.endswith('/result.json') and not Path(n).parts[1].startswith('.')}
    try:
        for result in results:
            sample, folder = result['id'], files[result['id']]
            bench = result.get('benchmark', {})
            f = bench.get('baseline_forwards', result['forwards'])
            length = len(result['answer_ids'])
            labels = result['labels']
            mapping = {'acoustic': 'audio_acoustic', 'semantic': 'audio_semantic', 'prompt': 'prompt'}
            def positions(group):
                if group == 'prefix':
                    return [i for i, label in enumerate(labels) if label != 'answer']
                return [i for i, label in enumerate(labels) if label == mapping[group]]
            prepared = SimpleNamespace(answer_start=len(labels) - length, group_positions=positions)
            from .protocol import conditions
            reference = row_budget(prepared, length, f, {'schedule': [{'group': 'prefix', 'refresh_every': 8}]}, 32)
            for spec in conditions(['freeze'], triple_refresh=['8:8:8', '16:8:16', '32:8:32', '0:8:0']):
                if not spec['condition'].startswith('freeze_triple_'):
                    continue
                hypothetical = row_budget(prepared, length, f, spec, 32)
                headroom_rows.append({'id': sample, 'condition': spec['condition'], 'forwards': f,
                    'shared8_token_layers': reference['computed_token_layers'],
                    'hypothetical_token_layers': hypothetical['computed_token_layers']})
            # Compare ALL schedules at the same baseline forward count, independently of their EOS changes.
            for policy in bench.get('policies', []):
                budget = row_budget(prepared, length, f, policy, 32)
                refresh = budget['refreshes_by_group']
                budget_rows.append({'id': sample, 'cohort': result.get('cohort'), 'condition': policy['condition'],
                    'prompt_tokens': result['prompt_tokens'], 'acoustic_tokens': result['audio_acoustic_tokens'],
                    'semantic_tokens': result['audio_semantic_tokens'], 'answer_slots': length,
                    'baseline_forwards': f, 'policy_forwards': policy['accounting'][0]['forwards'],
                    'same_input_computed_token_layers': budget['computed_token_layers'],
                    'same_input_dense_token_layers': budget['dense_token_layers'],
                    'same_input_refreshes_by_group_layer': json.dumps(refresh),
                    'decode_median_seconds': policy['median_seconds']})
            replay_name = folder + '/fixed_input_replay.json'
            if replay_name in names:
                replay = archive.read(replay_name)
                by_name = {p['condition']: p for p in replay['policies']}
                shared = by_name.get('freeze_prefix_r8')
                if shared:
                    for name, policy in by_name.items():
                        pairs[name].append((shared['median_seconds'], policy['median_seconds']))
            profile_name = folder + '/refresh_profile.json'
            if profile_name in names:
                profile = archive.read(profile_name)
                for item in profile['conditions']:
                    for phase, measured in item['phases'].items():
                        phase_rows.append({'id': sample, 'condition': item['condition'],
                                           'phase': phase, **measured})
                    cost_rows.append({'id': sample, 'condition': item['condition'],
                                      **{k: v for k, v in item['flop_estimate'].items() if k != 'note'},
                                      'computed_token_layers': item['row_budget']['computed_token_layers'],
                                      'forwards': profile['forwards']})
            drift_file = folder + '/drift.jsonl'
            attention_file = folder + '/attention_dynamics.jsonl'
            source = attention_file if attention_file in names else drift_file if drift_file in names else None
            if source is None:
                continue
            cutoff = result.get('baseline_content_revealed_step')
            content_count = result.get('first_eos_answer_position')
            if content_count is None:
                content_count = length
            content_positions, revealed, content_progress = set(range(content_count)), set(), {}
            steps_name = folder + '/steps.json'
            if steps_name in names:
                for event in archive.read(steps_name):
                    content_progress[event['step']] = (len(revealed & content_positions) / content_count
                                                       if content_count else None)
                    revealed.update(event['committed_positions'])
            # New diagnostics are already EOS-bounded. Exclude post-content-completion tail too.
            with archive.stream(source) as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['step'] == 0 or (cutoff is not None and row['step'] > cutoff):
                        continue
                    if source == drift_file:
                        if row['tensor'] != 'hidden' or row['reference'] != 'previous' or row['age'] != 1:
                            continue
                        if row['group'] not in {'audio_acoustic', 'audio_semantic', 'prompt'}:
                            continue
                        value = row['metrics']['cosine_distance']['mean']
                    else:
                        value = row['hidden_cosine_distance']
                    data = {'id': sample, 'cohort': result.get('cohort'), 'layer': row['layer'],
                            'group': row['group'], 'step': row['step'], 'block': row['block'],
                            'revealed_fraction': 1 - row['remaining_fraction'],
                            'block_revealed_fraction': 1 - row['active_remaining_fraction'],
                            'retrospective_content_revealed_fraction': row.get(
                                'retrospective_content_revealed_fraction', content_progress.get(row['step'])),
                            'previous_changed_tokens': row.get('previous_changed_tokens'),
                            'hidden_cosine_distance': value}
                    dynamics.append(data)
                    key = (sample, row['layer'], row['group'])
                    temporal_series[key].append(data)
                    for axis in ['revealed_fraction', 'block_revealed_fraction', 'retrospective_content_revealed_fraction']:
                        if data[axis] is None:
                            continue
                        bucket = min(9, int(data[axis] * 10))
                        progress_bins[(sample, row['layer'] // 8, row['group'], axis, bucket)].append(value)
                    if source == attention_file and row.get('unchanged_query_count', 0):
                        attention_rows.append({**data, **{k: row.get(k) for k in [
                            'attention_mass', 'attention_mass_abs_change', 'attention_l1_change',
                            'conditional_attention_js', 'contribution_relative_l2', 'query_cosine_distance']},
                            'token_spearman': row['token_drift_vs_attention_change']['spearman'],
                            'unchanged_query_count': row['unchanged_query_count'],
                            'unchanged_masked_query_count': row['unchanged_masked_query_count']})
        correlation_rows = []
        attention_series = defaultdict(list)
        for row in attention_rows:
            attention_series[(row['id'], row['layer'], row['group'])].append(row)
        for (sample, layer, group), values in temporal_series.items():
            for axis in ['revealed_fraction', 'block_revealed_fraction', 'retrospective_content_revealed_fraction']:
                selected = [v for v in values if v[axis] is not None]
                correlation_rows.append({'id': sample, 'layer': layer, 'group': group, 'comparison': axis,
                    **correlations([v['hidden_cosine_distance'] for v in selected], [v[axis] for v in selected])})
        for (sample, layer, group), values in attention_series.items():
            for axis in ['attention_l1_change', 'contribution_relative_l2', 'attention_mass_abs_change']:
                correlation_rows.append({'id': sample, 'layer': layer, 'group': group, 'comparison': axis,
                    **correlations([v['hidden_cosine_distance'] for v in values], [v[axis] for v in values])})
        bin_rows = [{'id': sample, 'layer_group': layer, 'group': group, 'axis': axis, 'bin': bucket,
                     'drift': statistics.mean(v), 'observations': len(v)}
                    for (sample, layer, group, axis, bucket), v in progress_bins.items()]
        headroom = {}
        for name in sorted({r['condition'] for r in headroom_rows}):
            values = [r for r in headroom_rows if r['condition'] == name]
            headroom[name] = sum(r['shared8_token_layers'] for r in values) / sum(r['hypothetical_token_layers'] for r in values)
        report = {'samples': len(results), 'token_counts': {k: summary([r[k] for r in results]) for k in
                  ['prompt_tokens', 'audio_acoustic_tokens', 'audio_semantic_tokens']},
                  'row_work_ratios_vs_shared8_NOT_latency_bounds': headroom,
                  'replay_speed_over_shared8': {name: {'samples': len(p),
                    'ratio_of_sums': sum(a for a, b in p) / sum(b for a, b in p)} for name, p in pairs.items()},
                  'drift_progress_rows': len(dynamics), 'attention_rows': len(attention_rows),
                  'notes': ['Row counts and estimated FLOPs are not wall-time bounds.',
                    'Shared-8 vs permanent-audio replay measures practical headroom in this backend, not a universal maximum.',
                    'All primary quality results retain baseline failures; inspect reasoning_report.md.',
                    'Correlations are computed separately within each sample/layer/group, not pooled across depths.',
                    'Global revealed fraction is coupled to forward index under fixed-quota decoding.',
                    'Retrospective content progress uses final pre-EOS positions; it is not an available online signal. '
                    'Bins may contain different examples; inspect the per-sample bin CSV.',
                    'Attention is reconstructed for sampled unchanged-identity answer queries. Query changes still contribute.',
                    'Representation/attention correlation is observational, not proof of causal importance.']}
        write_csv(out / 'token_forward_budget.csv', budget_rows)
        write_csv(out / 'hypothetical_row_headroom.csv', headroom_rows)
        write_csv(out / 'refresh_phase_timings.csv', phase_rows)
        write_csv(out / 'estimated_flop_budget.csv', cost_rows)
        write_csv(out / 'drift_progress.csv', dynamics)
        write_csv(out / 'drift_attention.csv', attention_rows)
        write_csv(out / 'diagnostic_correlations.csv', correlation_rows)
        write_csv(out / 'drift_progress_bins.csv', bin_rows)
        (out / 'diagnostic_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        lines = ['# Cache compute diagnostic', '', f"Completed examples: {len(results)}.", '',
                 'Continuous equal-input replay relative to shared-8:', '',
                 '| Condition | Paired examples | Speed over shared-8 |', '|---|---:|---:|']
        for name, item in report['replay_speed_over_shared8'].items():
            lines.append(f"| {name} | {item['samples']} | {item['ratio_of_sums']:.3f}x |")
        lines += ['', 'Token counts:']
        for name, item in report['token_counts'].items():
            lines.append(f"- {name}: min={item['min']}, median={item['median']}, max={item['max']}")
        lines += ['', 'Hypothetical row-work ratios vs shared-8, at each baseline forward count '
                  '(NOT measured latency or a latency upper bound):']
        for name, ratio in headroom.items():
            lines.append(f'- {name}: {ratio:.4f}x row-work ratio')
        lines += ['', *report['notes'], '', 'See refresh_profile.json per sample for synchronized cache-hit/refresh timing; '
                  'refresh_traces/index.json and traces provide separate operator attribution. '
                  'Initialization is reported separately; unavailable refresh phases are never extrapolated.']
        (out / 'diagnostic_report.md').write_text('\n'.join(lines) + '\n')
        if plots and bin_rows:
            plot_progress(out, bin_rows, attention_rows)
        print(f'Wrote cache/attention/progress diagnostics to {out}', flush=True)
        return report
    finally:
        archive.close()


def plot_progress(out, bins, attention):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['audio_acoustic', 'prompt', 'audio_semantic']
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for i, group in enumerate(groups):
        for j, axis in enumerate(['revealed_fraction', 'block_revealed_fraction', 'retrospective_content_revealed_fraction']):
            for depth in range(4):
                xs, ys = [], []
                for bucket in range(10):
                    values = [r['drift'] for r in bins if r['group'] == group and r['axis'] == axis
                              and r['layer_group'] == depth and r['bin'] == bucket]
                    if values:
                        xs.append((bucket + .5) / 10); ys.append(max(statistics.mean(values), 1e-12))
                axes[i, j].plot(xs, ys, marker='.', label=f'layers {depth*8}-{depth*8+7}')
            axes[i, j].set_yscale('log'); axes[i, j].set_ylabel(group + '\ncosine distance')
            axes[i, j].set_xlabel(axis.replace('_', ' '))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Baseline drift versus unmasking progress; equal sample means within bins')
    fig.tight_layout(); fig.savefig(out / 'drift_vs_unmasking.png', dpi=160); plt.close(fig)
    if attention:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        rng = __import__('random').Random(1729)
        for ax, group in zip(axes, groups):
            rows = [r for r in attention if r['group'] == group and r['attention_l1_change'] is not None]
            if len(rows) > 10000:
                rows = rng.sample(rows, 10000)
            im = ax.scatter([max(r['hidden_cosine_distance'], 1e-12) for r in rows],
                            [r['attention_l1_change'] for r in rows], c=[r['layer'] for r in rows],
                            s=3, alpha=.2, vmin=0, vmax=31)
            ax.set_xscale('log'); ax.set_title(group); ax.set_xlabel('Hidden cosine distance')
            ax.set_ylabel('Attention L1 change'); fig.colorbar(im, ax=ax, label='Layer')
        fig.tight_layout(); fig.savefig(out / 'drift_vs_attention.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    analyze(args.run, args.out, not args.no_plots)
