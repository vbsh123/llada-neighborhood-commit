"""Frozen semantic refresh ramp and symmetric fixed-prompt control."""
from copy import deepcopy

BOUNDARIES=(48,96)
PROVENANCE=dict(development_run='audio_independent_combined50_v22',samples=50,
    baseline_total_forwards=4688,mean_forwards=93.76,half_mean_forwards=46.88,
    phase_starts=[0,48,96],intervals=[8,16,32],
    rule='Ceil half-mean and mean to multiples of8; frozen before this experiment. Zero-based forwards, force refresh at phase entry.')


def conditions():
    def make(name,prompt=8,semantic=8,ramp=None):
        schedule=[dict(group=g,refresh_every=r) for g,r in
                  [('acoustic',16),('prompt',prompt),('semantic',semantic)]]
        if ramp:
            s=next(s for s in schedule if s['group']==ramp)
            s['refresh_phases']=[dict(start=t,refresh_every=r) for t,r in zip((0,*BOUNDARIES),(8,16,32))]
        return dict(condition=name,mode='freeze',schedule=deepcopy(schedule),
                    ramp_group=ramp,phase_provenance=PROVENANCE if ramp else None)
    return [make('ramp_control_a16_p8_s8'),
            make('ramp_semantic_a16_p8_s8x16x32',ramp='semantic'),
            make('ramp_prompt_a16_p8x16x32_s8',ramp='prompt'),
            make('ramp_constant_a16_p8_s16',semantic=16),
            make('ramp_constant_a16_p8_s32',semantic=32),
            make('ramp_constant_a16_p16_s8',prompt=16),
            make('ramp_constant_a16_p32_s8',prompt=32)]


def analyze(run):
    import csv,json
    from pathlib import Path
    run=Path(run);out=run/'analysis';out.mkdir(exist_ok=True)
    secondary={(r['id'],r['condition']):r for r in csv.DictReader((out/'numeric_secondary_pairs.csv').open())}
    rows=[]
    for path in sorted((run/'samples').glob('*/result.json')):
        if path.parent.name.startswith('.'):continue
        r=json.loads(path.read_text());policies=r['benchmark']['policies']
        common=next(p for p in policies if p['condition']=='ramp_control_a16_p8_s8')
        for p in policies:
            a=p['accounting'][0];n=a['forwards'];s=secondary[r['id'],p['condition']]
            if p.get('ramp_group') and min(n,common['accounting'][0]['forwards'])<=BOUNDARIES[0]:
                if p['visible_answer_ids']!=common['visible_answer_ids'] or n!=common['accounting'][0]['forwards']:
                    raise ValueError('Ramp changed output before its first phase transition')
            counts=a['refresh_counts_by_schedule_and_layer']
            refresh={spec['group']:counts[i][0] for i,spec in enumerate(p['schedule'])}
            rows.append(dict(id=r['id'],condition=p['condition'],ramp_group=p.get('ramp_group'),
                acoustic_tokens=r.get('audio_acoustic_tokens'),semantic_tokens=r.get('audio_semantic_tokens'),
                prompt_tokens=r.get('prompt_tokens'),
                correct=s['correct']=='True',seconds=p['median_seconds'],tokens=p['visible_answer_tokens'],
                forwards=n,entered_phase16=bool(p.get('ramp_group') and n>48),
                entered_phase32=bool(p.get('ramp_group') and n>96),
                acoustic_refreshes=refresh['acoustic'],prompt_refreshes=refresh['prompt'],semantic_refreshes=refresh['semantic'],
                computed_token_layers=a['computed_token_layers']))
    lines=['# Development refresh ramps','',
        'Scheduled8->16->32 reuse at zero-based forwards48/96, frozen from development mean93.76. '
        'Prompt control applies the identical schedule to fixed prompt text, never answer tokens. '
        'Acoustic interval16, untargeted group8. No pruning, boosting, attention probes or drift. '
        'Group token counts differ: equal schedules are not equal compute budgets. '
        'All examples stay included; phase-entry subsets are diagnostics only.','',
        '| Policy | N | Correct | TPS | Entered16 /32 | Mean S / P refreshes |',
        '|---|---:|---:|---:|---:|---:|']
    for spec in conditions():
        v=[r for r in rows if r['condition']==spec['condition']]
        if not v:continue
        n=len(v);tps=sum(r['tokens'] for r in v)/sum(r['seconds'] for r in v)
        lines.append(f"| {spec['condition']} | {n} | {sum(r['correct'] for r in v)} | {tps:.2f} | "
            f"{sum(r['entered_phase16'] for r in v)} / {sum(r['entered_phase32'] for r in v)} | "
            f"{sum(r['semantic_refreshes'] for r in v)/n:.2f} / {sum(r['prompt_refreshes'] for r in v)/n:.2f} |")
    (out/'cache_ramp_report.md').write_text('\n'.join(lines)+'\n')
    (out/'cache_ramp_provenance.json').write_text(json.dumps(PROVENANCE,indent=2)+'\n')
    with (out/'cache_ramp_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\n'.join(lines),flush=True)
    return rows


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run)
