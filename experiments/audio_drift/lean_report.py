"""Aggregate single-pass decode TPS, original and frozen secondary accuracy."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def analyze(run):
    run=Path(run);out=run/'analysis';out.mkdir(exist_ok=True)
    secondary={(r['id'],r['condition']):r for r in csv.DictReader((out/'numeric_secondary_pairs.csv').open())}
    rows=[];cohorts=set()
    for path in sorted((run/'samples').glob('*/result.json')):
        if path.parent.name.startswith('.'):continue
        r=json.loads(path.read_text());b=r['benchmark']
        cohorts.add(r.get('cohort','unspecified'))
        if b.get('protocol')!='single-pass-eos-v1':raise ValueError('Expected lean single-pass protocol')
        policies=[dict(condition='baseline',score=r['score'],median_seconds=b['baseline_median_seconds'],
            visible_answer_tokens=b['baseline_visible_answer_tokens'],accounting=[dict(forwards=b['baseline_forwards'])])]+b['policies']
        for p in policies:
            s=secondary[r['id'],p['condition']]
            plan=p.get('pruning',{});picks=p['accounting'][0].get('pruning',{}).get('selections',[])
            rows.append(dict(id=r['id'],condition=p['condition'],original_correct=bool(p['score']['correct']),
                correct=s['correct']=='True',parse_failed=s['parse_failed']=='True',prediction=s['prediction'],reference=s['reference'],
                seconds=p['median_seconds'],visible_answer_tokens=p['visible_answer_tokens'],forwards=p['accounting'][0]['forwards'],
                threshold=plan.get('threshold_uniform'),fraction=plan.get('fraction'),control=plan.get('control'),
                prune_layer_start=plan.get('layer_start'),
                activated=bool(picks),removed_acoustic=sum(v['dropped_by_group']['acoustic'] for v in picks),
                removed_semantic=sum(v['dropped_by_group']['semantic'] for v in picks),
                removed_prompt=sum(v['dropped_by_group']['prompt'] for v in picks)))
    grouped=defaultdict(list)
    for row in rows:grouped[row['condition']].append(row)
    title='Development pruning sensitivity' if 'development_v24' in cohorts else 'Held-out pruning'
    lines=[f'# {title}: single-pass accuracy and throughput','',
        'One decode per condition/example after process-wide warmup; baseline first and each audio source precedes its reference-guided generic. '
        'TPS is summed visible answer tokens excluding EOS divided by summed synchronized decode time. '
        'Decode includes pruning observations, selection and commitment; excludes audio preparation/scoring/file writes. '
        'No repeated per-example timing or identical-input replay. Legacy median fields hold one observation. '
        'Frozen secondary final-answer scores preserve original scores. All examples, including baseline failures, remain included.','',
        '| Policy | N | Original correct | Secondary correct | Parse failures | Forwards | Tokens/sec | Decode speedup |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    base=sum(r['seconds'] for r in grouped['baseline'])
    for name,v in grouped.items():
        seconds=sum(r['seconds'] for r in v);tokens=sum(r['visible_answer_tokens'] for r in v)
        lines.append(f"| {name} | {len(v)} | {sum(r['original_correct'] for r in v)} | {sum(r['correct'] for r in v)} | "
            f"{sum(r['parse_failed'] for r in v)} | {sum(r['forwards'] for r in v)} | {tokens/seconds:.2f} | {base/seconds:.3f}x |")
    names=list(grouped)
    for i,left in enumerate(names):
        a={r['id']:r for r in grouped[left]}
        for right in names[i+1:]:
            if left!='baseline' and (grouped[left][0]['threshold'],grouped[left][0]['fraction'])!=(grouped[right][0]['threshold'],grouped[right][0]['fraction']):
                continue
            pairs=[(a[r['id']],r) for r in grouped[right]]
            wins=sum(not a['correct'] and b['correct'] for a,b in pairs)
            losses=sum(a['correct'] and not b['correct'] for a,b in pairs)
            lines+=['',f'{right} versus {left}: {wins} recoveries, {losses} losses (frozen secondary scores).']
    (out/'lean_report.md').write_text('\n'.join(lines)+'\n')
    with (out/'lean_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\n'.join(lines),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run)
