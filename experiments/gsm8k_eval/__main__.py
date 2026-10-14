"""Re-score saved GSM8K result.json files; never imports torch or generates text."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from .scoring import COMMIT, load_protocol, score_saved


def read_json(path):
    raw=path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def discover_modes(run,requested=None):
    if (run/'samples').is_dir():
        modes={run.name:run}
    else:
        modes={p.name:p for p in sorted(run.iterdir()) if p.is_dir() and (p/'samples').is_dir()}
    if requested:
        missing=set(requested)-set(modes)
        if missing: raise ValueError(f'Unknown modes: {sorted(missing)}; found {sorted(modes)}')
        modes={name:modes[name] for name in requested}
    if not modes: raise ValueError(f'No saved sample directories in {run}')
    return modes


def load_results(directory):
    results={}
    for path in sorted((directory/'samples').glob('*/result.json')):
        record,digest=read_json(path)
        sid=str(record['sample_id'])
        if sid!=path.parent.name or sid in results:
            raise ValueError(f'Duplicate/mismatched sample ID: {path}')
        if any(not isinstance(record.get(k),str) for k in ('answer','question','reference')):
            raise ValueError(f'Missing answer/question/reference text: {path}')
        results[sid]={'result':record,'path':str(path.resolve()),'sha256':digest}
    return results


def saved_samples(run,modes):
    candidates=[run/'samples.jsonl']
    if len(modes)==1 and next(iter(modes.values()))==run:
        candidates.append(run.parent/'samples.jsonl')
    for path in candidates:
        if path.exists():
            raw=path.read_bytes()
            rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            samples={str(r['id']):r for r in rows}
            if len(samples)!=len(rows): raise ValueError('Duplicate IDs in saved samples.jsonl')
            return samples,{'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest()}
    return None,None


def paired(a,b):
    counts=dict(both_correct=0,both_wrong=0,baseline_only_correct=0,treatment_only_correct=0)
    for ca,cb in zip(a,b):
        key='both_correct' if ca and cb else 'both_wrong' if not ca and not cb else 'baseline_only_correct' if ca else 'treatment_only_correct'
        counts[key]+=1
    return {'accuracy_change_percentage_points':100*(sum(b)-sum(a))/len(a),'paired_correctness':counts}


def evaluate(run,*,out=None,modes=None,baseline_mode=None,allow_incomplete=False):
    run=run.resolve()
    paths=discover_modes(run,modes)
    data={name:load_results(path) for name,path in paths.items()}
    samples,samples_info=saved_samples(run,paths)
    expected=set(samples) if samples is not None else set.union(*(set(d) for d in data.values()))
    common=set.intersection(*(set(d) for d in data.values()))
    if not common: raise ValueError('No jointly completed samples to score')
    if any(set(records)-expected for records in data.values()):
        raise ValueError('Results contain IDs outside saved sample manifest')
    if not allow_incomplete and any(set(records)!=expected for records in data.values()):
        raise ValueError('Incomplete or differing sample sets. Wait for completion, or explicitly use --allow-incomplete for the common completed subset.')
    ids=sorted(common)
    for sid in ids:
        records=[items[sid]['result'] for items in data.values()]
        if any((r['question'],r['reference'])!=(records[0]['question'],records[0]['reference']) for r in records):
            raise ValueError(f'Question/reference differs between modes for {sid}')
        if samples is not None:
            sample=samples[sid]
            if (sample['question'],sample['answer'])!=(records[0]['question'],records[0]['reference']):
                raise ValueError(f'Result question/reference differs from saved sample {sid}')
    if baseline_mode is None:
        baseline_mode=next((name for name in ('baseline','active_baseline','top1','original') if name in paths),next(iter(paths)))
    if baseline_mode not in paths: raise ValueError('Requested baseline mode not found')
    provenance,_,_=load_protocol()
    summary={'protocol':'lm_eval_gsm8k_v3.0_saved_answers_v1','harness_commit':COMMIT,
        'primary_metric':'exact_match,flexible-extract','secondary_metric':'exact_match,strict-match',
        'run':str(run),'samples_scored_per_mode':len(ids),'sample_ids':ids,
        'expected_samples':len(expected),'allow_incomplete':allow_incomplete,
        'completed_counts':{mode:len(d) for mode,d in data.items()},
        'missing_result_ids_by_mode':{mode:sorted(expected-set(d)) for mode,d in data.items()},
        'excluded_completed_ids_by_mode':{mode:sorted(set(d)-common) for mode,d in data.items()},
        'saved_samples':samples_info,'baseline_mode':baseline_mode,'modes':{},'paired_comparisons':{},
        'generation_changed':False,'official_dataset_content_independently_verified':False,
        'note':'Scoring-only: original prompts, shots, stop rules and outputs are unchanged. This is not a claim of reproducing the harness default five-shot generation protocol.'}
    per_sample=[]; scores={}; inputs=[]
    for mode,items in data.items():
        rr=[items[sid]['result'] for sid in ids]
        scores[mode]=score_saved([r['answer'] for r in rr],[r['reference'] for r in rr])
        legacy=[r.get('correct_numeric') for r in rr]
        known=[v for v in legacy if isinstance(v,bool)]
        mode_summary={'metrics':{},'legacy_numeric_accuracy':sum(known)/len(known) if known else None,
                      'legacy_scored_samples':len(known)}
        for metric,values in scores[mode].items():
            changed=[sid for sid,old,new in zip(ids,legacy,values['correct']) if isinstance(old,bool) and old!=new]
            mode_summary['metrics'][f'exact_match,{metric}']={
                'accuracy':values['accuracy'],'invalid_extraction_count':values['invalid_count'],
                'disagreements_with_legacy_count':len(changed),'disagreements_with_legacy_ids':changed}
        summary['modes'][mode]=mode_summary
        for index,sid in enumerate(ids):
            record=rr[index]
            row={'mode':mode,'sample_id':sid,'question':record['question'],'reference':record['reference'],
                 'answer':record['answer'],'legacy_correct_numeric':legacy[index],
                 'legacy_predicted_answer':record.get('predicted_answer'),
                 'legacy_answer_extraction':record.get('answer_extraction'),
                 'result_path':items[sid]['path'],'result_sha256':items[sid]['sha256']}
            for name,values in scores[mode].items():
                row[name+'_prediction']=values['predictions'][index]
                row[name+'_correct']=values['correct'][index]
            per_sample.append(row)
            inputs.append({'mode':mode,'sample_id':sid,'path':row['result_path'],'sha256':row['result_sha256']})
    for mode in paths:
        if mode==baseline_mode: continue
        summary['paired_comparisons'][f'{baseline_mode}_vs_{mode}']={
            f'exact_match,{metric}':paired(scores[baseline_mode][metric]['correct'],scores[mode][metric]['correct'])
            for metric in scores[baseline_mode]}
    out=(out or run/'gsm8k_harness_eval').resolve()
    if out==run or any(out==p.resolve() or (p/'samples').resolve() in out.parents or out==(p/'samples').resolve() for p in paths.values()):
        raise ValueError('Use a dedicated evaluation output directory, outside source sample directories')
    out.mkdir(parents=True,exist_ok=True)
    summary['output_directory']=str(out)
    def write(name,text):
        path=out/name; partial=path.with_suffix(path.suffix+'.tmp')
        partial.write_text(text,encoding='utf-8'); partial.replace(path)
    write('per_sample.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in per_sample))
    with (out/'per_sample.csv.tmp').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_sample[0]))
        writer.writeheader(); writer.writerows(per_sample)
    (out/'per_sample.csv.tmp').replace(out/'per_sample.csv')
    write('provenance.json',json.dumps({'upstream':provenance,'inputs':inputs,
        'scorer_source_sha256':hashlib.sha256((Path(__file__).parent/'scoring.py').read_bytes()).hexdigest(),
        'evaluator_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'created_unix':time.time()},indent=2)+'\n')
    write('summary.json',json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,action='append',help='Run root or a single saved mode; repeat for multiple runs')
    p.add_argument('--out',type=Path,help='Dedicated evaluator folder; only with one --run')
    p.add_argument('--modes',nargs='+',help='Default: discover all saved modes')
    p.add_argument('--baseline-mode',help='Mode used for paired comparisons')
    p.add_argument('--allow-incomplete',action='store_true',help='Explicitly score only the shared completed sample IDs')
    return p


def main():
    args=parser().parse_args()
    if args.out and len(args.run)!=1: raise ValueError('--out requires exactly one --run')
    if len({p.resolve() for p in args.run})!=len(args.run): raise ValueError('Duplicate run paths')
    for run in args.run:
        result=evaluate(run,out=args.out,modes=args.modes,baseline_mode=args.baseline_mode,
                        allow_incomplete=args.allow_incomplete)
        print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
