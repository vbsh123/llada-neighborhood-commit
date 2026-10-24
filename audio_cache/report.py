"""Paired final-answer accuracy, throughput and latency reports."""
import csv
import json
from .storage import write_json

def summarize(out):
    rows = []
    for path in sorted((out/'samples').glob('*/result.json')):
        if path.parent.name.startswith('.'):
            continue
        sample = json.loads(path.read_text())
        for policy in [sample['baseline'], *sample['policies']]:
            rows.append(dict(id=sample['id'], condition=policy['condition'], correct=policy['secondary_correct'],
                original_correct=policy['score']['correct'], parse_failed=policy['secondary_parse_failed'],
                seconds=policy['seconds'], tokens=policy['visible_answer_tokens'],
                forwards=policy['accounting'][0]['forwards']))
    if not rows:
        return
    base = {r['id']:r for r in rows if r['condition']=='baseline'}
    lines = ['# Audio inference development comparison', '',
        'Frozen secondary final-answer accuracy; original scores retained. One synchronized EOS decode per '
        'condition/example after process warmup. TPS counts visible answer tokens excluding EOS. '
        'All examples stay included. Decoding remains 512 answer slots, block32, maximum256 forwards. '
        'No drift recording or replay. Cache initialization and row gathers/scatters are included in timing.', '',
        '| Policy | N | Correct | Accuracy | Δ baseline (pp) | TPS | Δ TPS | Decode speedup | Wins / losses |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for name in sorted({r['condition'] for r in rows}, key=lambda n:(n!='baseline',n)):
        selected = [r for r in rows if r['condition']==name]
        paired = [base[r['id']] for r in selected]
        seconds = sum(r['seconds'] for r in selected)
        base_seconds = sum(r['seconds'] for r in paired)
        tps = sum(r['tokens'] for r in selected)/seconds
        base_tps = sum(r['tokens'] for r in paired)/base_seconds
        n = len(selected)
        correct = sum(r['correct'] for r in selected)
        delta = 100*(correct-sum(r['correct'] for r in paired))/n
        wins = sum(r['correct'] and not b['correct'] for r,b in zip(selected,paired))
        losses = sum(b['correct'] and not r['correct'] for r,b in zip(selected,paired))
        tps_change = 100*(tps/base_tps-1) if base_tps else None
        tps_label = f'{tps_change:+.1f}%' if tps_change is not None else 'N/A'
        summary[name] = dict(samples=n, correct=correct, accuracy=correct/n, accuracy_delta_pp=delta,
            tps=tps, tps_change_percent=tps_change, decode_speedup=base_seconds/seconds,
            wins=wins, losses=losses, parse_failures=sum(r['parse_failed'] for r in selected))
        lines.append(f'| {name} | {n} | {correct} | {100*correct/n:.1f}% | {delta:+.1f} | '
                     f'{tps:.2f} | {tps_label} | {base_seconds/seconds:.3f}× | {wins} / {losses} |')
    analysis = out/'analysis'
    analysis.mkdir(exist_ok=True)
    (analysis/'report.md').write_text('\n'.join(lines)+'\n')
    with (analysis/'pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out/'summary.json', summary)
    return '\n'.join(lines)
