"""Offline analysis: no model download or GPU required."""
import argparse
import csv
import gzip
import json
from pathlib import Path

from .core import groups, nearest
from .run import dump


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def word_distance(position, anchors, word_map):
    if not word_map.get('valid') or not anchors:
        return None
    mapping = word_map['token_word_indices']
    current = mapping[position]
    previous = [w for p in anchors for w in mapping[p]]
    return min((abs(a-b) for a in current for b in previous), default=None)


def event_row(sample_id, step, row, result, kind):
    distance_words = word_distance(row['position'], step['previous_commits'], result.get('word_map', {}))
    end = result['answer_token_length']
    valid = row['position'] < end and not row['special']
    # Primary event denominator also requires an in-answer nonspecial anchor.
    anchors_valid = bool(step['previous_commits']) and all(
        p < end and not result['token_dictionary'].get(str(result['final_ids'][p]), {}).get('special', False)
        for p in step['previous_commits'])
    answer_candidates = [r for r in step['positions'] if r['eligible'] and not r['special'] and r['position'] < end]
    answer_local = [r for r in answer_candidates if nearest(r['position'], step['previous_commits'])['nonlocal'] is False]
    return {'sample_id': sample_id, 'step': step['step'], 'kind': kind,
            'position': row['position'], 'token_id': row['token_id'], 'text': row['text'],
            'category': row['category'], 'special': row['special'],
            'confidence': row['confidence'], 'entropy': row['entropy'], 'margin': row['margin'],
            'delta_confidence': row['delta_confidence'], 'same_token_delta': row['same_token_delta'],
            'prediction_changed': row['prediction_changed'],
            **nearest(row['position'], step['previous_commits']),
            'previous_commits': step['previous_commits'], 'word_distance': distance_words,
            'word_nonlocal': distance_words > 1 if distance_words is not None else None,
            'resolved_fraction': step['resolved_fraction'],
            'normalized_position': row['position']/max(1, len(result['final_ids'])-1),
            'distance_from_frontier': row['position']-step['metrics']['left_frontier'],
            'block_start': step['block_start'], 'block_end': step['block_end'],
            'block_transition': step.get('block_transition', False),
            'eligible': row['eligible'], 'before_final_stop': row['position'] < end,
            'primary_population': valid and anchors_valid,
            'local_option_available': step['metrics'].get('local_option_available'),
            'local_candidate_count': step['metrics'].get('local_candidate_count'),
            'uniform_choice_nonlocal_rate': step['metrics'].get('uniform_choice_nonlocal_rate'),
            'answer_local_option_available': bool(answer_local) if anchors_valid else None,
            'best_answer_local_confidence': max((r['confidence'] for r in answer_local), default=None),
            'matches_final_token': row['token_id'] == result['final_ids'][row['position']],
            'correct_numeric': result.get('correct_numeric', result['correct_lenient']),
            'correct_strict': result['correct_strict'], 'correct_lenient': result['correct_lenient']}


def bootstrap_mean(values):
    import numpy as np
    if not values:
        return {'mean': None, 'ci95': None, 'n_problems': 0}
    arr = np.array(values)
    rng = np.random.default_rng(1729)
    estimates = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(1000)])
    return {'mean': float(arr.mean()), 'ci95': np.quantile(estimates, [.025, .975]).tolist(), 'n_problems': len(arr)}


def sample_plot(steps, result, path):
    import numpy as np
    import matplotlib.pyplot as plt
    n, length = len(steps), len(result['final_ids'])
    conf = np.full((n, length), np.nan)
    delta = conf.copy()
    commit_x, commit_y = [], []
    for j, step in enumerate(steps):
        for row in step['positions']:
            conf[j, row['position']] = row['confidence']
            if row['delta_confidence'] is not None:
                delta[j, row['position']] = row['delta_confidence']
        commit_x.extend(step['commit_positions'])
        commit_y.extend([j]*len(step['commit_positions']))
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
    for ax, data, title, cmap, lo, hi in [
        (axes[0], conf, 'Top candidate probability at every still-masked position', 'viridis', 0, 1),
        (axes[1], delta, 'Probability change since previous forward pass (token may change)', 'coolwarm', -1, 1)]:
        im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap, vmin=lo, vmax=hi)
        ax.scatter(commit_x, commit_y, s=3, color='red', label='committed')
        ax.set_title(title)
        ax.set_ylabel('Forward step')
        fig.colorbar(im, ax=ax, shrink=.8)
    axes[2].scatter(commit_x, commit_y, s=8)
    if all(len(s['commit_positions']) == 1 for s in steps):
        axes[2].plot(commit_x, commit_y, linewidth=.6, alpha=.5)
    axes[2].invert_yaxis()
    axes[2].set_title('Commit trajectory: lines connect successive single-token choices')
    axes[2].set_ylabel('Forward step')
    for ax in axes:
        ax.set_xlabel('Response token position (zero-based)')
        if result['first_stop_position'] is not None:
            ax.axvline(result['first_stop_position'], color='orange', linestyle='--', label='final first stop')
    fig.suptitle(f"Sample {result['sample_id']} | numeric match={result.get('correct_numeric', result['correct_lenient'])} | red dots=commits")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def overview(events, per_problem, out):
    import numpy as np
    import matplotlib.pyplot as plt
    primary = [e for e in events if e['kind'] == 'commit' and e['primary_population']]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    if primary:
        axes[0, 0].hist([e['signed_distance'] for e in primary], bins=41)
        axes[0, 0].set_xlabel('Signed distance from nearest previous-batch fill')
        axes[0, 0].set_ylabel('Committed tokens')
        axes[0, 0].set_title('Where choices go: negative=left, positive=right')
        grouped = {}
        for e in primary:
            key = (e['sample_id'], min(9, int(e['resolved_fraction']*10)))
            grouped.setdefault(key, []).append(e)
        xs, ys = [], []
        for b in range(10):
            rates = []
            for sid in per_problem:
                bucket = grouped.get((sid, b), [])
                if bucket: rates.append(sum(e['nonlocal'] for e in bucket)/len(bucket))
            xs.append((b+.5)/10)
            ys.append(np.mean(rates) if rates else np.nan)
        axes[0, 1].plot(xs, ys, marker='o')
        axes[0, 1].set(xlabel='Fraction of response positions already filled', ylabel='Mean per-problem nonlocal fraction', ylim=(0, 1))
        categories = sorted({e['category'] for e in primary})
        values, labels = [], []
        for cat in categories:
            cc = [e for e in primary if e['category'] == cat]
            values.append(sum(e['nonlocal'] for e in cc)/len(cc))
            labels.append(f'{cat} (n={len(cc)})')
        axes[1, 0].barh(labels, values)
        axes[1, 0].set(xlabel='Nonlocal fraction among commits of this category', xlim=(0, 1))
        jumps = [e for e in primary if e['nonlocal']]
        axes[1, 1].scatter([e['resolved_fraction'] for e in jumps], [e['normalized_position'] for e in jumps], s=4, alpha=.35)
        axes[1, 1].set(xlabel='Fraction already filled', ylabel='Destination / response length', title='When and where nonlocal choices occur')
    else:
        for ax in axes.flat:
            ax.text(.5, .5, 'No primary-population commits', ha='center', transform=ax.transAxes)
    fig.savefig(out / 'overview.png', dpi=150)
    plt.close(fig)


def analyze_run(run, out, max_plots, rise):
    out.mkdir(parents=True, exist_ok=True)
    events, step_rows, threshold_rows, problem_rows, example_candidates = [], [], [], [], []
    per_problem = {}
    paths = sorted((run / 'samples').glob('*/result.json'))
    if not paths:
        raise ValueError(f'No completed samples in {run}')
    for index, result_path in enumerate(paths):
        result = json.loads(result_path.read_text())
        steps = [r for r in records(result_path.with_name('trace.jsonl.gz')) if r['type'] == 'step']
        sid = result['sample_id']
        sample_events = []
        last_block = None
        for step in steps:
            step['block_transition'] = last_block is not None and step['block_start'] != last_block
            last_block = step['block_start']
            metrics = step['metrics']
            step_rows.append({'sample_id': sid, 'step': step['step'], 'resolved_fraction': step['resolved_fraction'],
                              'block_transition': step['block_transition'], 'forward_seconds': step['forward_seconds'],
                              'commit_count': len(step['commit_positions']),
                              **{k: v for k, v in metrics.items() if k != 'thresholds'}})
            for stat in metrics['thresholds']:
                threshold_rows.append({'sample_id': sid, 'step': step['step'], **stat})
            population = [r for r in step['positions'] if r['eligible'] and not r['special'] and r['position'] < result['answer_token_length']]
            for threshold in sorted({s['threshold'] for s in metrics['thresholds']}):
                above = [r for r in population if r['confidence'] >= threshold]
                islands = groups(r['position'] for r in above)
                threshold_rows.append({'sample_id': sid, 'step': step['step'], 'scope': 'answer_eligible_nonspecial',
                                       'threshold': threshold, 'population': len(population), 'count': len(above),
                                       'fraction': len(above)/len(population) if population else None,
                                       'island_count': len(islands), 'islands': [[g[0], g[-1]] for g in islands],
                                       'nonlocal_count': sum(nearest(r['position'], step['previous_commits'])['nonlocal'] is True for r in above)
                                                        if step['previous_commits'] else None})
            for row in step['positions']:
                if row['committed']:
                    sample_events.append(event_row(sid, step, row, result, 'commit'))
                if row['eligible'] and row['nonlocal'] and row['delta_confidence'] is not None and row['delta_confidence'] >= rise:
                    event = event_row(sid, step, row, result, 'remote_rise')
                    sample_events.append(event)
                    if event['primary_population']:
                        example_candidates.append((event, result['question']))
                if row['position'] == metrics['leader_position']:
                    sample_events.append(event_row(sid, step, row, result, 'leader'))
        events.extend(sample_events)
        primary = [e for e in sample_events if e['kind'] == 'commit' and e['primary_population']]
        optional_jumps = [e for e in primary if e['answer_local_option_available']]
        words = [e for e in primary if e['word_nonlocal'] is not None]
        rises = [e for e in sample_events if e['kind'] == 'remote_rise' and e['primary_population']]
        pp = {'sample_id': sid, 'correct_strict': result['correct_strict'], 'correct_lenient': result['correct_lenient'],
              'correct_numeric': result.get('correct_numeric', result['correct_lenient']),
              'prompt_protocol': result.get('prompt_protocol', 'legacy_reasoning_and_marker'),
              'steps': result['steps'], 'hit_length_limit': result['hit_length_limit'],
              'primary_commits': len(primary), 'nonlocal_commits': sum(e['nonlocal'] for e in primary),
              'nonlocal_fraction': sum(e['nonlocal'] for e in primary)/len(primary) if primary else None,
              'commits_with_local_alternative': len(optional_jumps),
              'nonlocal_fraction_with_local_alternative': sum(e['nonlocal'] for e in optional_jumps)/len(optional_jumps) if optional_jumps else None,
              'word_nonlocal_fraction': sum(e['word_nonlocal'] for e in words)/len(words) if words else None,
              'remote_rise_count': len(rises),
              'remote_rise_final_match_fraction': sum(e['matches_final_token'] for e in rises)/len(rises) if rises else None}
        problem_rows.append(pp)
        per_problem[sid] = pp
        if index < max_plots:
            sample_plot(steps, result, out / f'sample_{sid}.png')
    write_csv(out / 'events.csv', events)
    write_csv(out / 'steps.csv', step_rows)
    write_csv(out / 'thresholds.csv', threshold_rows)
    write_csv(out / 'problems.csv', problem_rows)
    summary = {'run': str(run), 'samples': len(paths), 'rise_threshold': rise,
               'primary_definition': 'Eligible nonspecial commit before final first stop; every previous-batch anchor also nonspecial and before stop. Nonlocal means nearest anchor distance >1 token.',
               'nonlocal_fraction_problem_bootstrap': bootstrap_mean([p['nonlocal_fraction'] for p in problem_rows if p['nonlocal_fraction'] is not None]),
               'nonlocal_with_local_alternative_problem_bootstrap': bootstrap_mean([p['nonlocal_fraction_with_local_alternative'] for p in problem_rows if p['nonlocal_fraction_with_local_alternative'] is not None]),
               'word_nonlocal_fraction_problem_bootstrap': bootstrap_mean([p['word_nonlocal_fraction'] for p in problem_rows if p['word_nonlocal_fraction'] is not None]),
               'strict_accuracy': sum(p['correct_strict'] for p in problem_rows)/len(paths),
               'numeric_accuracy': sum(p['correct_numeric'] for p in problem_rows)/len(paths),
               'primary_accuracy_metric': 'numeric_accuracy',
               'strict_accuracy_note': 'Marker-only diagnostic; not primary accuracy for question-only prompts',
               'prompt_protocols': sorted({p['prompt_protocol'] for p in problem_rows}),
               'length_limit_fraction': sum(p['hit_length_limit'] for p in problem_rows)/len(paths),
               'remote_rise_events': sum(p['remote_rise_count'] for p in problem_rows)}
    dump(out / 'summary.json', summary)
    # Largest increases selected mechanically; include successes and failures without hand-picking.
    ranked = sorted(example_candidates, key=lambda pair: (-pair[0]['delta_confidence'], pair[0]['sample_id'], pair[0]['step'], pair[0]['position']))[:30]
    dump(out / 'examples.json', [{'event': event, 'question': question} for event, question in ranked])
    overview(events, per_problem, out)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-plots', type=int, default=8)
    p.add_argument('--rise', type=float, default=.15, help='Absolute increase in top candidate probability')
    args = p.parse_args()
    if args.rise < 0 or args.max_plots < 0:
        p.error('rise and max-plots must be nonnegative')
    summaries = []
    for index, run in enumerate(args.run):
        out = args.out if len(args.run) == 1 else args.out / f'{index}_{run.name}'
        summaries.append(analyze_run(run, out, args.max_plots, args.rise))
    if len(summaries) > 1:
        dump(args.out / 'comparison.json', summaries)


if __name__ == '__main__':
    main()
