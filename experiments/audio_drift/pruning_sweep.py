"""Development-only one-factor sensitivity sweep; no held-out tuning."""
from .progressive_pruning import progressive_conditions


SETTINGS=((.5,.25),(.25,.25),(1.,.25),(.5,.125),(.5,.5))


def conditions(args):
    result=[]
    for threshold,fraction in SETTINGS:
        result.extend(s for s in progressive_conditions(args.length,args.block_length,args.steps,
            args.acoustic_start,args.progressive_semantic_start,args.progressive_interval,
            args.progressive_patience,threshold,fraction,args.seed)
            if s['pruning']['control'] in ('audio','generic'))
    return result


def analyze(run):
    import csv
    from pathlib import Path
    run=Path(run);out=run/'analysis'
    with (out/'lean_pairs.csv').open() as f:rows=list(csv.DictReader(f))
    baseline=[r for r in rows if r['condition']=='baseline']
    lines=['# Development pruning sensitivity','',
        'Same previously inspected50 questions; one-factor sweep. No held-out conclusions. '
        'Final-answer scores use the unchanged frozen secondary extractor. TPS includes pruning decision overhead. '
        'Generic matches each audio source\'s event times/counts and ranks its own full prefix; autonomous EOS can truncate later events.','',
        f"Baseline: {sum(r['correct']=='True' for r in baseline)}/{len(baseline)} correct; "
        f"{sum(int(r['visible_answer_tokens']) for r in baseline)/sum(float(r['seconds']) for r in baseline):.2f} TPS.",'',
        '| Threshold / uniform | Remove fraction | Audio correct | Generic correct | Audio wins / losses | Audio TPS | Generic TPS |',
        '|---|---|---:|---:|---:|---:|---:|']
    details=[]
    for threshold,fraction in SETTINGS:
        chosen={c:[r for r in rows if r['control']==c and float(r['threshold'])==threshold and float(r['fraction'])==fraction]
                for c in ('audio','generic')}
        audio,generic=chosen['audio'],chosen['generic']
        lookup={r['id']:r for r in generic}
        if len(audio)!=len(baseline) or len(lookup)!=len(baseline):raise ValueError('Incomplete sensitivity matrix')
        wins=sum(a['correct']=='True' and lookup[a['id']]['correct']!='True' for a in audio)
        losses=sum(a['correct']!='True' and lookup[a['id']]['correct']=='True' for a in audio)
        tps=lambda v:sum(int(r['visible_answer_tokens']) for r in v)/sum(float(r['seconds']) for r in v)
        ac=sum(a['correct']=='True' for a in audio);gc=sum(g['correct']=='True' for g in generic)
        lines.append(f'| {threshold:g} | {fraction:.1%} | {ac}/{len(audio)} | {gc}/{len(generic)} | {wins} / {losses} | {tps(audio):.2f} | {tps(generic):.2f} |')
        for control,v in chosen.items():
            details.append(dict(threshold=threshold,fraction=fraction,control=control,samples=len(v),
                correct=sum(r['correct']=='True' for r in v),tps=tps(v),
                activated=sum(r['activated']=='True' for r in v),
                mean_removed_acoustic=sum(int(r['removed_acoustic']) for r in v)/len(v),
                mean_removed_semantic=sum(int(r['removed_semantic']) for r in v)/len(v),
                mean_removed_prompt=sum(int(r['removed_prompt']) for r in v)/len(v)))
    lines+=['','Inspect sensitivity across nearby settings rather than selecting the highest development accuracy as validation. '
        'Start indices16/44, check interval8 and patience2 stay fixed. This sweep does not independently tune the generic method.']
    (out/'sensitivity_report.md').write_text('\n'.join(lines)+'\n')
    with (out/'sensitivity_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print('\n'.join(lines),flush=True)
    return details


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run)
