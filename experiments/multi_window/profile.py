"""Replay one saved trace on Vast: no new answers, no change to saved results.

Compare the original selective backend against the optimized one and vanilla on
identical original inputs. This is a model-call diagnostic, not end-to-end TPS.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

from confidence_geography.run import load_model, dump
from confidence_geography.core import select
from experiments.token_sleep.benchmark import probability_comparison, compact_rows
from experiments.token_sleep.selective import SelectiveForward

REFERENCE_REVISION = '377223c'
ROOT = Path(__file__).resolve().parents[2]


def reference_backend():
    source = subprocess.check_output(['git','show',
        f'{REFERENCE_REVISION}:experiments/token_sleep/selective.py'],cwd=ROOT,text=True)
    scope = {'__name__':'experiments.token_sleep._profile_reference',
             '__package__':'experiments.token_sleep'}
    exec(compile(source,'<pinned pre-audit selective.py>','exec'),scope)
    return scope['SelectiveForward'],hashlib.sha256(source.encode()).hexdigest()


def read_trace(path, max_steps):
    with gzip.open(path,'rt') as handle:
        records = [json.loads(line) for line in handle]
    header = records[0]
    if header.get('type') != 'header':
        raise ValueError('Trace must begin with header')
    steps = [r for r in records if r.get('type')=='step'][:max_steps]
    if not steps or [r['step'] for r in steps] != list(range(len(steps))):
        raise ValueError('Require a contiguous trace beginning at step zero')
    if steps[0]['sleep_positions']:
        raise ValueError('Initial forward must initialize every cached position')
    config = header['config']
    state = [config['mask_id']]*config['length']
    frames = []
    for record in steps:
        sleeping = set(record['sleep_positions'])
        masked = {p for p,t in enumerate(state) if t==config['mask_id']}
        if not sleeping <= masked:
            raise ValueError('Expected persistent-region trace: only MASKs sleep')
        awake = sorted(masked-sleeping)
        positions,ids = record['commit_positions'],record['commit_ids']
        if not awake or len(positions)!=len(ids) or not set(positions)<=set(awake):
            raise ValueError('Invalid saved commitment geometry')
        frames.append({'state':list(state),'awake':awake,'sleepers':sorted(sleeping),
                       'step':record['step']})
        for p,tid in zip(positions,ids):
            if tid==config['mask_id']: raise ValueError('Saved commitment is MASK')
            state[p]=tid
    return header,frames


def forward(model,backend,x,pl,frame):
    positions=[pl+p for p in frame['awake']]
    if backend is None:
        full=model(x,use_cache=False)
        logits=full.logits[:,positions]
        del full
        return logits
    return backend(x,[pl+p for p in frame['sleepers']],positions)


def aggregate(rows):
    result={}
    for mode in ('original','legacy_selective','updated_selective'):
        selected=[r for r in rows if r['mode']==mode]
        result[mode]={}
        for kind in ('all','all_awake','sparse'):
            subset=[r for r in selected if kind=='all' or r['kind']==kind]
            if subset:
                result[mode][kind]={'calls':len(subset),
                    'mean_wall_ms':1000*sum(r['wall_seconds'] for r in subset)/len(subset),
                    'mean_cuda_ms':sum(r['cuda_ms'] for r in subset)/len(subset)}
    return result


def run(args):
    import torch
    if min(args.steps,args.repeats)<1:
        raise ValueError('Positive steps/repeats required')
    manifest=json.loads((args.run/'manifest.json').read_text())
    samples=[json.loads(line) for line in (args.run/'samples.jsonl').read_text().splitlines()]
    sample_id=args.sample_id or samples[0]['id']
    path=args.run/'multi_window'/'samples'/sample_id/'trace.jsonl.gz'
    header,frames=read_trace(path,args.steps)
    if not any(f['sleepers'] for f in frames):
        raise ValueError('Choose more steps or another sample: no sleeping geometry')
    legacy,reference_hash=reference_backend()
    config=manifest['config']
    if header['config']!=config:
        raise ValueError('Trace and manifest config mismatch')
    args.out.mkdir(parents=True,exist_ok=False)
    print('Loading pinned model for SAME-INPUT TRACE REPLAY only; no generation.',flush=True)
    model,tokenizer=load_model(config)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    pl=len(header['prompt_ids'])
    calls=[]; comparisons=[]
    modes=('original','legacy_selective','updated_selective')
    with torch.inference_mode():
        initial=torch.tensor([header['prompt_ids']+frames[0]['state']],device=model.device)
        for _ in range(2):
            model(initial,use_cache=False)
            legacy(model)(initial,[],list(range(pl,initial.shape[1])))
            SelectiveForward(model)(initial,[],list(range(pl,initial.shape[1])))
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            backends={'original':None,'legacy_selective':legacy(model),
                      'updated_selective':SelectiveForward(model)}
            order=modes[repeat%3:]+modes[:repeat%3]
            for frame in frames:
                x=torch.tensor([header['prompt_ids']+frame['state']],device=model.device)
                outputs={}
                for mode in order:
                    torch.cuda.synchronize()
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    t0=time.perf_counter(); begin.record()
                    logits=forward(model,backends[mode],x,pl,frame)
                    end.record(); end.synchronize()
                    calls.append({'repeat':repeat,'step':frame['step'],'mode':mode,
                        'kind':'sparse' if frame['sleepers'] else 'all_awake',
                        'active_rows':x.shape[1]-len(frame['sleepers']) if mode!='original' else x.shape[1],
                        'prediction_rows':len(frame['awake']),
                        'wall_seconds':time.perf_counter()-t0,'cuda_ms':begin.elapsed_time(end)})
                    if mode!='original': outputs[mode]=logits
                    del logits
                cmp=probability_comparison(outputs['updated_selective'],outputs['legacy_selective'],config['mask_id'])
                decisions={}
                for mode,logits in outputs.items():
                    rows=compact_rows(logits,frame['awake'],config['mask_id'])
                    chosen=select(rows,config['policy'],1,config['commit_threshold'])
                    decisions[mode]=[(r['position'],r['token_id']) for r in chosen]
                comparisons.append({'repeat':repeat,'step':frame['step'],**cmp,
                    'selection_equal':decisions['updated_selective']==decisions['legacy_selective']})
                del outputs
            print(f'Replay {repeat+1}/{args.repeats} done.',flush=True)
            del backends
        if args.capture:
            # One representative local forward; rebuild its caches by replaying
            # all preceding original inputs. Do not include profiler overhead in timings.
            target=next(i for i,f in enumerate(frames) if f['sleepers'])
            backends={'original':None,'legacy_selective':legacy(model),
                      'updated_selective':SelectiveForward(model)}
            for frame in frames[:target]:
                x=torch.tensor([header['prompt_ids']+frame['state']],device=model.device)
                for mode in modes:
                    forward(model,backends[mode],x,pl,frame)
            frame=frames[target]
            x=torch.tensor([header['prompt_ids']+frame['state']],device=model.device)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                for mode in modes:
                    with torch.profiler.record_function(mode):
                        forward(model,backends[mode],x,pl,frame)
                        torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.out/'operators.json'))
            (args.out/'operators.txt').write_text(prof.key_averages().table(
                sort_by='self_cuda_time_total',row_limit=60))
    max_error=max(c['max_probability_difference'] for c in comparisons)
    report={'scope':'One saved question, teacher-forced original states; model calls only, NOT generated quality or end-to-end TPS',
        'source_run':str(args.run.resolve()),'sample_id':sample_id,'steps':len(frames),
        'repeats':args.repeats,'gpu':torch.cuda.get_device_name(0),
        'reference_revision':REFERENCE_REVISION,'reference_source_sha256':reference_hash,
        'torch_version':torch.__version__,'timing':aggregate(calls),
        'old_vs_new_max_probability_difference':max_error,
        'old_vs_new_winner_changes':sum(c['winner_changed_count'] for c in comparisons),
        'old_vs_new_selection_changes':sum(not c['selection_equal'] for c in comparisons),
        'old_vs_new_checks':comparisons,'calls':calls}
    dump(args.out/'profile.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('calls','old_vs_new_checks')},indent=2),flush=True)
    if max_error>.02 or report['old_vs_new_winner_changes'] or report['old_vs_new_selection_changes']:
        print('WARNING: old/new execution differs; inspect parity before interpreting performance.',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sample-id')
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--capture',action='store_true',help='Also capture separate untimed CPU/CUDA operator trace')
    return p


if __name__=='__main__': run(parser().parse_args())
