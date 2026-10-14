"""Real token-sleep quality/latency benchmark. CUDA execution belongs on Vast."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

from confidence_geography.core import select
from confidence_geography.run import (MODEL, REVISION, PROMPT_PROTOCOL, SCORING_PROTOCOL,
    dump, emit, load_model, score_logits, score_gsm8k, run_lock, command)
from .run import load_samples
from .simulation import FrozenStates, SleepSchedule, checked_blocks
from .selective import SelectiveForward

MODES = ('active_baseline', 'real_sleep')
HERE = Path(__file__).resolve().parent


def compact_rows(logits, response_positions, mask_id):
    rows = score_logits(logits[0], list(range(len(response_positions))), {}, mask_id, 2)
    for row, position in zip(rows,response_positions):
        row.update(position=position,eligible=True)
    return rows


def prompt_ids(tokenizer, question):
    text = tokenizer.apply_chat_template([{'role':'user','content':question}],
                                         tokenize=False,add_generation_prompt=True)
    return tokenizer.encode(text,add_special_tokens=False)


def checked_input(model, tokenizer, question, length, mask_id):
    import torch
    prompt = prompt_ids(tokenizer,question)
    limit = getattr(model.config,'max_position_embeddings',None) or getattr(model.config,'max_sequence_length',None)
    if mask_id in prompt or (limit and len(prompt)+length > limit):
        raise ValueError('MASK in prompt or context limit exceeded')
    return torch.tensor([prompt+[mask_id]*length],device=model.device),len(prompt)


def probability_comparison(actual, reference, mask_id):
    """Chunked full-vocabulary probability/logit error; no GPU-sized FP32 copy."""
    import torch
    result = {'max_probability_difference':0.,'max_logit_difference':0.,'winner_changed_count':0,
              'positions':actual.shape[1]}
    for start in range(0,actual.shape[1],16):
        a,b = actual[0,start:start+16].float(),reference[0,start:start+16].float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite validation logits')
        pa,pb = a.softmax(-1),b.softmax(-1)
        result['max_probability_difference'] = max(result['max_probability_difference'],(pa-pb).abs().max().item())
        result['max_logit_difference'] = max(result['max_logit_difference'],(a-b).abs().max().item())
        pa[:,mask_id] = -1; pb[:,mask_id] = -1
        result['winner_changed_count'] += (pa.argmax(-1)!=pb.argmax(-1)).sum().item()
    return result


def validate(model, tokenizer, sample, config):
    """Teacher-forced identical inputs, real versus dense and frozen-state simulator.

    Cache updates from comparison forwards must not touch simulator caches. The
    original model supplies commits, so every variant sees exactly the same input.
    A separate timer treats a teacher-forced committed sleeper as freshly awake.
    """
    import torch
    x,pl = checked_input(model,tokenizer,sample['question'],config['length'],config['mask_id'])
    blocks = checked_blocks(model)
    dense, selective = SelectiveForward(model),SelectiveForward(model)
    schedule = SleepSchedule(config['length'],'simulated_sleep',confidence=config['sleep_confidence'],
        high_entropy=config['high_entropy'],high_sleep=config['high_sleep'],low_sleep=0)
    report = {'steps':[],'scope':'identical teacher-forced original inputs',
              'max_probability_tolerance':config['validation_tolerance'],'passed':True}
    with torch.inference_mode(),FrozenStates(blocks) as simulator:
        for step in range(min(config['validation_steps'],config['length'])):
            state = x[0,pl:].tolist()
            masked = [p for p,t in enumerate(state) if t==config['mask_id']]
            if not masked: break
            sleepers = schedule.prepare(step,state,config['mask_id'])
            absolute = [pl+p for p in masked]
            awake = [p for p in masked if p not in set(sleepers)]
            with simulator.bypass():
                raw = model(x,use_cache=False).logits
                all_active = dense(x,[],absolute)
            raw_masked = raw[:,absolute]
            raw_rows = compact_rows(raw_masked,masked,config['mask_id'])
            dense_rows = compact_rows(all_active,masked,config['mask_id'])
            base_cmp = probability_comparison(all_active,raw_masked,config['mask_id'])
            base_selection_equal = [(r['position'],r['token_id']) for r in select(raw_rows,config['policy'],1,config['commit_threshold'])] == [(r['position'],r['token_id']) for r in select(dense_rows,config['policy'],1,config['commit_threshold'])]
            simulator.begin([pl+p for p in sleepers])
            sim = model(x,use_cache=False).logits
            simulator.end()
            with simulator.bypass():
                actual = selective(x,[pl+p for p in sleepers],[pl+p for p in awake])
            sim_awake = sim[:,[pl+p for p in awake]]
            sleep_cmp = probability_comparison(actual,sim_awake,config['mask_id'])
            actual_rows = compact_rows(actual,awake,config['mask_id'])
            sim_rows = compact_rows(sim_awake,awake,config['mask_id'])
            sleep_selection_equal = [(r['position'],r['token_id']) for r in select(actual_rows,config['policy'],1,config['commit_threshold'])] == [(r['position'],r['token_id']) for r in select(sim_rows,config['policy'],1,config['commit_threshold'])]
            passed = (base_cmp['max_probability_difference'] <= config['validation_tolerance']
                      and sleep_cmp['max_probability_difference'] <= config['validation_tolerance']
                      and base_selection_equal and sleep_selection_equal)
            report['steps'].append({'step':step,'sleepers':sleepers,'dense_vs_original':base_cmp,
                'selective_vs_simulator':sleep_cmp,'dense_selection_equal':base_selection_equal,
                'sleep_selection_equal':sleep_selection_equal,'passed':passed})
            report['passed'] &= passed
            # Original commitments are fixed for this validation replay only.
            chosen = select(raw_rows,config['policy'],1,config['commit_threshold'])
            schedule.finish(step,actual_rows,[r['position'] for r in chosen])
            for row in chosen: x[0,pl+row['position']] = row['token_id']
            del raw,sim,all_active,actual,raw_masked,sim_awake
    report['tested_sleep_forwards'] = sum(bool(s['sleepers']) for s in report['steps'])
    # Arbitrary sparse geometry is checked even if the uncertainty rule did not
    # select sleepers in these early forwards.
    x,pl = checked_input(model,tokenizer,sample['question'],config['length'],config['mask_id'])
    sparse = SelectiveForward(model)
    with torch.inference_mode(),FrozenStates(blocks) as simulator:
        simulator.begin([]); model(x,use_cache=False); simulator.end()
        with simulator.bypass(): sparse(x,[],list(range(pl,pl+config['length'])))
        positions = list(range(0,config['length'],3))
        # Update one nonsleeping token to create a context change.
        if config['length'] > 1:
            with simulator.bypass(): raw = model(x,use_cache=False).logits
            winner = compact_rows(raw[:,[pl+1]],[1],config['mask_id'])[0]['token_id']
            x[0,pl+1] = winner
            del raw
        sleepers = [pl+p for p in positions]
        awake = [pl+p for p in range(config['length']) if p not in set(positions)]
        if awake:
            simulator.begin(sleepers); sim = model(x,use_cache=False).logits; simulator.end()
            with simulator.bypass(): actual = sparse(x,sleepers,awake)
            sparse_cmp = probability_comparison(actual,sim[:,awake],config['mask_id'])
            report['sparse_original_position_check'] = sparse_cmp
            report['passed'] &= sparse_cmp['max_probability_difference'] <= config['validation_tolerance']
        else:
            report['sparse_original_position_check'] = {'skipped':'length leaves no awake response row'}
    return report


def decode(model,tokenizer,sample,config,mode,directory):
    import torch
    x,pl = checked_input(model,tokenizer,sample['question'],config['length'],config['mask_id'])
    backend = None if mode=='original' else SelectiveForward(model)
    schedule = SleepSchedule(config['length'],'simulated_sleep' if mode=='real_sleep' else 'baseline',
        confidence=config['sleep_confidence'],high_entropy=config['high_entropy'],high_sleep=config['high_sleep'],low_sleep=0)
    # Equal untimed warm-up count. Reset all state afterward.
    warm_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(config['warmup_forwards']):
            if backend: backend(x,[],list(range(pl,pl+config['length'])))
            else: model(x,use_cache=False)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-warm_start
    if backend: backend = SelectiveForward(model)
    trace = []
    model_seconds = 0.
    slept = masked_total = fallbacks = active_total = head_total = 0
    step = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        while True:
            state = x[0,pl:].tolist()
            masked = [p for p,t in enumerate(state) if t==config['mask_id']]
            if not masked: break
            sleepers = schedule.prepare(step,state,config['mask_id'])
            sleeping = set(sleepers)
            awake = [p for p in masked if p not in sleeping]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if backend:
                logits = backend(x,[pl+p for p in sleepers],[pl+p for p in awake])
                active_count,head_count = backend.last_active_count,backend.last_head_count
            else:
                full = model(x,use_cache=False)
                logits = full.logits[:,[pl+p for p in awake]]
                active_count = head_count = x.shape[1]
                del full
            torch.cuda.synchronize(); seconds = time.perf_counter()-t0
            model_seconds += seconds
            rows = compact_rows(logits,awake,config['mask_id'])
            chosen = select(rows,config['policy'],1,config['commit_threshold'])
            if not chosen: raise RuntimeError('No fresh candidate after fallback')
            assigned = schedule.finish(step,rows,[r['position'] for r in chosen])
            for row in chosen: x[0,pl+row['position']] = row['token_id']
            slept += len(sleepers); masked_total += len(masked)
            fallbacks += int(schedule.fallback)
            active_total += active_count; head_total += head_count
            trace.append({'step':step,'masked_count':len(masked),'sleep_positions':sleepers,
                'active_rows':active_count,'head_rows':head_count,'model_seconds':seconds,
                'all_asleep_full_wake':schedule.fallback,'assigned':assigned,
                'commit_positions':[r['position'] for r in chosen],
                'commit_ids':[r['token_id'] for r in chosen]})
            step += 1
            del logits
            if step > config['length']: raise RuntimeError('Failed to make progress')
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    tokens = x[0,pl:].tolist()
    stops = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    vocab = tokenizer.get_vocab()
    if '<|eot_id|>' in vocab: stops.add(vocab['<|eot_id|>'])
    end = min((p for p,t in enumerate(tokens) if t in stops),default=config['length'])
    answer = tokenizer.decode(tokens[:end],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    result = {'sample_id':sample['id'],'mode':mode,'question':sample['question'],
        'reference':sample['answer'],'answer':answer,'final_ids':tokens,'steps':step,
        'prompt_protocol':PROMPT_PROTOCOL,'scoring_protocol':SCORING_PROTOCOL,
        **score_gsm8k(answer,sample),'first_stop_position':end if end<len(tokens) else None,
        'hit_length_limit':end==len(tokens),'decode_seconds':elapsed,'model_seconds':model_seconds,
        'warmup_seconds_excluded':warmup_seconds,'warmup_forwards_excluded':config['warmup_forwards'],
        'sleep_position_forwards':slept,'masked_position_forwards':masked_total,
        'active_transformer_rows':active_total,'vocabulary_head_rows':head_total,
        'all_asleep_full_wakes':fallbacks,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    # All serialization happens after the measured decode interval.
    with gzip.open(directory/'trace.jsonl.gz.partial','wt',compresslevel=3) as handle:
        emit(handle,{'type':'header','config':config,'mode':mode,'sample':sample,'prompt_ids':x[0,:pl].tolist()})
        for record in trace: emit(handle,{'type':'step',**record})
        emit(handle,{'type':'result',**result})
    (directory/'trace.jsonl.gz.partial').replace(directory/'trace.jsonl.gz')
    dump(directory/'result.json',result)
    return result


def evaluate(out,samples,modes=MODES):
    results = {mode:[json.loads((out/mode/'samples'/s['id']/'result.json').read_text()) for s in samples]
               for mode in modes}
    summary = {'samples':len(samples),'modes':{},'comparisons':{},
        'timing_scope':'Full response including post-stop slots; decode loop includes scoring/timers/cache updates; excludes warmup, validation, model loading and serialization.'}
    for mode,rr in results.items():
        summary['modes'][mode] = {'numeric_accuracy':sum(r['correct_numeric'] for r in rr)/len(rr),
            **{key:sum(r[key] for r in rr) for key in ('steps','decode_seconds','model_seconds',
            'sleep_position_forwards','masked_position_forwards','active_transformer_rows',
            'vocabulary_head_rows','all_asleep_full_wakes')}}
        mm=summary['modes'][mode]
        mm['sleep_fraction_of_masked_position_forwards']=mm['sleep_position_forwards']/mm['masked_position_forwards']
    for baseline,treatment in [('original','active_baseline'),('active_baseline','real_sleep'),('original','real_sleep')]:
        if baseline not in results or treatment not in results:
            continue
        a,b = summary['modes'][baseline],summary['modes'][treatment]
        counts = dict(both_correct=0,both_wrong=0,baseline_only_correct=0,treatment_only_correct=0)
        for ra,rb in zip(results[baseline],results[treatment]):
            ca,cb = ra['correct_numeric'],rb['correct_numeric']
            key = 'both_correct' if ca and cb else 'both_wrong' if not ca and not cb else 'baseline_only_correct' if ca else 'treatment_only_correct'
            counts[key] += 1
        summary['comparisons'][f'{baseline}_vs_{treatment}'] = {
            'decode_speedup':a['decode_seconds']/b['decode_seconds'],
            'model_speedup':a['model_seconds']/b['model_seconds'],
            'transformer_row_reduction_fraction':1-b['active_transformer_rows']/a['active_transformer_rows'],
            'vocabulary_head_row_reduction_fraction':1-b['vocabulary_head_rows']/a['vocabulary_head_rows'],
            'accuracy_change_percentage_points':100*(b['numeric_accuracy']-a['numeric_accuracy']),
            'paired_correctness':counts}
    dump(out/'comparison.json',summary)
    return summary


def fingerprint():
    base = HERE.parents[1]/'confidence_geography'
    paths = [*HERE.glob('*.py'),base/'run.py',base/'core.py']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(paths))).hexdigest()


def run(args):
    import torch
    import numpy as np
    if min(args.samples,args.length,args.validation_steps)<1 or min(args.offset,args.warmup_forwards,args.high_sleep)<0:
        raise ValueError('Invalid counts')
    if not 0 < args.validation_tolerance <= .02 or not 0 <= args.commit_threshold <= 1:
        raise ValueError('Invalid validation/commit tolerance')
    SleepSchedule(args.length,'simulated_sleep',confidence=args.sleep_confidence,
                  high_entropy=args.high_entropy,high_sleep=args.high_sleep,low_sleep=0)
    config = {k:v for k,v in vars(args).items() if k!='out'}
    config['samples_from'] = str(args.samples_from.resolve()) if args.samples_from else None
    modes = ('original',*MODES) if args.include_original else MODES
    config.update(model=MODEL,revision=REVISION,mask_id=126336,modes=list(modes),experiment='real_token_sleep_v1')
    out = args.out; out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        manifest_path = out/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['config']!=config or manifest['source_sha256']!=fingerprint():
                raise ValueError('Arguments/source changed; use a new --out')
            source = (out/'samples.jsonl').read_bytes()
            if hashlib.sha256(source).hexdigest()!=manifest['samples_sha256']:
                raise ValueError('Saved samples changed')
            samples = [json.loads(s) for s in source.decode().splitlines()]
        else:
            if any(out.glob('*/samples/*/result.json')): raise ValueError('Results without manifest')
            samples,info = load_samples(args)
            source = ''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in samples).encode()
            (out/'samples.jsonl').write_bytes(source)
            dump(manifest_path,{'config':config,'dataset_info':info,'samples_sha256':hashlib.sha256(source).hexdigest(),
                'source_sha256':fingerprint(),'git_commit':command('git','rev-parse','HEAD'),
                'created_unix':time.time(),'torch_version':torch.__version__,
                'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        pending = []
        for index,sample in enumerate(samples):
            # Rotate order to avoid always measuring sleeping last on a warmer GPU.
            rotation = index%len(modes)
            order = modes[rotation:]+modes[:rotation]
            for mode in order:
                directory = out/mode/'samples'/sample['id']
                if not (directory/'result.json').exists(): pending.append((sample,mode,directory))
                elif not (directory/'trace.jsonl.gz').exists(): raise ValueError('Missing completed trace')
        if pending:
            description = 'layer diagnostic only; no generation or timing' if args.diagnose_only else f'{len(samples)} matched questions × {len(modes)} modes {modes}'
            print(f'Loading pinned LLaDA: {description}; full window={args.length}.',flush=True)
            model,tokenizer = load_model(config)
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            if args.diagnose_only:
                from .diagnostics import diagnose
                report=diagnose(model,tokenizer,samples[0],config)
                dump(out/'layer_diagnostics.json',report)
                print(json.dumps({key:value for key,value in report.items() if key!='layers'},indent=2),flush=True)
                return
            report = validate(model,tokenizer,samples[0],config)
            dump(out/'validation.json',report)
            print('GPU validation:',json.dumps(report),flush=True)
            if not report['passed']: raise RuntimeError('GPU numerical/selection parity failed; inspect validation.json before benchmarking')
            for sample,mode,directory in pending:
                random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
                directory.mkdir(parents=True,exist_ok=True)
                result = decode(model,tokenizer,sample,config,mode,directory)
                print(f"DONE mode={mode} sample={sample['id']} correct={result['correct_numeric']} forwards={result['steps']} decode_seconds={result['decode_seconds']:.3f}",flush=True)
            del model
        print(json.dumps(evaluate(out,samples,modes),indent=2),flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=10)
    p.add_argument('--samples-from',type=Path)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--policy',choices=['top1','threshold'],default='top1')
    p.add_argument('--include-original',action='store_true',help='Optionally rerun full vanilla timing; default compares active_baseline and real_sleep only')
    p.add_argument('--diagnose-only',action='store_true',help='Local layer parity in native dtype versus FP32; no generations or benchmark')
    p.add_argument('--commit-threshold',type=float,default=.9)
    p.add_argument('--sleep-confidence',type=float,default=.5)
    p.add_argument('--high-entropy',type=float,default=2.)
    p.add_argument('--high-sleep',type=int,default=2)
    p.add_argument('--warmup-forwards',type=int,default=2)
    p.add_argument('--validation-steps',type=int,default=6)
    p.add_argument('--validation-tolerance',type=float,default=.02,
                   help='Maximum absolute probability error; validation also requires matching selected commits')
    return p


if __name__=='__main__': run(parser().parse_args())
