"""Combine the three overnight reports without mixing historical/current timing."""
import argparse
import csv
import json
from pathlib import Path

from .storage import write_json


def summarize(root):
    configs=[]
    combined={}
    pairs=[]
    for suite in ('cache','prune','sparse'):
        path=root/suite
        if not (path/'summary.json').exists():
            continue
        config=json.loads((path/'config.json').read_text())
        configs.append(config)
        if config['settings']['baseline_mode']!='saved':
            raise ValueError('Overnight report expects saved-baseline mode')
        if configs[0]['input_sha256']!=config['input_sha256'] or configs[0]['inputs']!=config['inputs']:
            raise ValueError('Overnight suites do not use identical ordered questions')
        for key in ('source_sha256','model_revisions','decoder','torch'):
            if config[key]!=configs[0][key]:
                raise ValueError('Overnight suites use different model/decoder/source/environment settings')
        summary=json.loads((path/'summary.json').read_text())
        for name,record in summary.items():
            if name=='baseline':
                if 'baseline' not in combined:
                    combined[name]=dict(suite='saved_reference',**record)
                continue
            combined[name]=dict(suite=suite,**record)
        with (path/'analysis'/'pairs.csv').open() as stream:
            pairs.extend(dict(suite=suite,**row) for row in csv.DictReader(stream) if row['condition']!='baseline')
    lines=['# Overnight audio inference study','',
        'Same development questions and frozen answer scorer. Baseline outputs/TPS are historical; '
        'all seven experimental conditions are measured on this GPU. No uncached baseline is generated. '
        'TPS counts visible answer tokens, so answer length and EOS/forward counts may differ. '
        'Observation, anchor search, selection and gather/scatter overhead are included.','',
        '| Suite | Policy | N | Correct | Accuracy | Δ saved baseline (pp) | TPS | Timing source |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for name,record in combined.items():
        lines.append(f"| {record['suite']} | {name} | {record['samples']} | {record['correct']} | "
            f"{100*record['accuracy']:.1f}% | {record['accuracy_delta_pp']:+.1f} | {record['tps']:.2f} | {record['timing_source']} |")
    control=combined.get('cache_a32_p32_s32')
    sparse=combined.get('sparse_audio_cache323232')
    if control and sparse:
        a={r['id']:r for r in pairs if r['condition']=='cache_a32_p32_s32'}
        b={r['id']:r for r in pairs if r['condition']=='sparse_audio_cache323232'}
        ids=sorted(set(a)&set(b))
        wins=sum(a[i]['correct']=='False' and b[i]['correct']=='True' for i in ids)
        losses=sum(a[i]['correct']=='True' and b[i]['correct']=='False' for i in ids)
        if ids:
            base_tps=sum(int(a[i]['tokens']) for i in ids)/sum(float(a[i]['seconds']) for i in ids)
            sparse_tps=sum(int(b[i]['tokens']) for i in ids)/sum(float(b[i]['seconds']) for i in ids)
            gain=f'{100*(sparse_tps/base_tps-1):+.1f}%' if base_tps else 'N/A'
            lines+=['',f'Sparse+cache32 versus ordinary cache32 on {len(ids)} paired questions: '
                     f'{wins} recoveries / {losses} losses; TPS change {gain}. '
                     'Separate sequential phases on the same GPU; this is not fixed-input replay.']
    output=root/'analysis'
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'summary.json',combined)
    if pairs:
        with (output/'pairs.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    summarize(parser.parse_args().root)
