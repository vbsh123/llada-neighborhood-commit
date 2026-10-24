"""Fixed-prefix cache schedules on the original50; use saved baseline by default."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import torch

from .cache import PrefixCache
from .checkout import DEFAULT_ROOT, COMMIT, verify_checkout
from .data import read_manifest, load_audio, fingerprint
from .decode import decode
from .extractor import extract_answer, VERSION
from .model import load_model, prepare, DIFFA_REPO, DIFFA_REVISION, LLM_REPO, LLM_REVISION
from .report import summarize
from .scoring import score, canonical_number
from .storage import write_json

WHISPER_REVISION='06f233fe06e710322aca913c1bc4249a0d71fce1'
POLICIES={'888':(8,8,8),'16816':(16,8,16),'161616':(16,16,16),
          '323232':(32,32,32),'321632':(32,16,32),'323216':(32,32,16)}


def specifications(args):
    if args.suite=='cache':
        return [dict(condition='cache_a%d_p%d_s%d'%POLICIES[name],intervals=POLICIES[name]) for name in args.policies]
    if args.suite=='prune':
        return [dict(condition='prune_global_'+control,pruning=dict(control=control)) for control in ('audio','generic')]
    sparse=dict(chunk_size=32,dense_layers=8,layer_group=8,anchor_fractions=[.5,.25,.125],proxy_queries=128,neighbors=1)
    return [dict(condition='sparse_audio',sparse=sparse),
            dict(condition='sparse_audio_cache323232',intervals=(32,32,32),sparse=sparse)]


def resolve_policy(spec,completed):
    resolved=deepcopy(spec)
    if resolved.get('pruning',{}).get('control')=='generic':
        resolved['pruning']['source']=completed['prune_global_audio']['accounting'][0]['pruning']
    return resolved


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['cache','prune','sparse'],default='cache')
    p.add_argument('--manifest',type=Path,default=Path('data/development50.jsonl'))
    p.add_argument('--out',type=Path,default=Path('runs/cache161616_50'))
    p.add_argument('--policies',choices=POLICIES,nargs='+',default=['161616'])
    p.add_argument('--baseline-mode',choices=['saved','fresh'],default='saved')
    p.add_argument('--reuse-run',type=Path,help='Reuse completed, matching cache results from an interrupted v28 run')
    p.add_argument('--samples',type=int,default=50)
    p.add_argument('--diffa-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--verify-samples',type=int,default=0)
    p.add_argument('--progress-every',type=int,default=32)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=1729)
    return p


@torch.no_grad()
def generation(llm,prepared,tokenizer,args,intervals=None,*,policy=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    cache,sparse=None,None
    with ExitStack() as stack:
        if policy and policy.get('pruning'):
            from .pruning import GlobalPruning
            if intervals:
                raise ValueError('Pruning is evaluated separately from temporal caching')
            cache=stack.enter_context(GlobalPruning(llm,prepared,**policy['pruning']))
        elif intervals:
            cache=stack.enter_context(PrefixCache(llm,prepared,intervals))
        if policy and policy.get('sparse'):
            from .sparse import AudioSparseAttention
            sparse=stack.enter_context(AudioSparseAttention(llm,prepared,512,cache=cache,**policy['sparse']))
        final,events=decode(llm,prepared,512,32,256,
            eos_id=tokenizer.eos_token_id,progress_every=args.progress_every)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    account=cache.accounting() if cache else dict(forwards=len(events))
    if sparse:
        account['sparse']=sparse.accounting()
    ids=final[0,prepared.prompt.shape[1]:].tolist()
    eos=tokenizer.eos_token_id
    visible=ids[:ids.index(eos)] if eos in ids else ids
    return final,dict(seconds=elapsed,visible_answer_tokens=len(visible),visible_answer_ids=visible,
        answer_has_no_eos=eos not in ids,peak_allocated_bytes=peak,
        text=tokenizer.decode(visible,skip_special_tokens=True).strip(),accounting=[account])


def grade(measurement,name,target):
    prediction,method=extract_answer(measurement['text'])
    return dict(**measurement,condition=name,score=score(measurement['text'],target),
        secondary_prediction=prediction,secondary_correct=prediction==canonical_number(target),
        secondary_parse_failed=prediction is None,extraction=method)


@torch.no_grad()
def main():
    args=parser().parse_args()
    if (not 1<=args.samples<=50 or args.cpu_threads<1 or min(args.verify_samples,args.progress_every)<0
            or len(set(args.policies))!=len(args.policies)):
        raise ValueError('Invalid sample/thread/verification/policy settings')
    if args.baseline_mode=='saved' and args.verify_samples:
        raise ValueError('Saved-baseline mode does not run additional uncached verification decodes')
    if args.reuse_run and args.suite!='cache':
        raise ValueError('Cross-run reuse is supported only for cache-only experiments')
    specs=specifications(args)
    verify_checkout(args.diffa_root)
    all_rows=read_manifest(args.manifest)
    rows=all_rows[:args.samples]
    saved=json.loads(Path(__file__).with_name('saved_baselines.json').read_text())
    if saved['development50_sha256']!=fingerprint(all_rows) or set(saved['measurements'])!={r['id'] for r in all_rows}:
        raise ValueError('Saved baselines do not match this cohort')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run on a BF16-capable CUDA GPU')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=hashlib.sha256()
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.suffix in ('.py','.json'):
            source.update(path.name.encode())
            source.update(path.read_bytes())
    config=dict(protocol='standalone-audio-inference-v30',specifications=specs,
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        inputs=rows,development50_sha256=fingerprint(all_rows),source_sha256=source.hexdigest(),
        upstream_commit=COMMIT,torch=torch.__version__,scorer=VERSION,
        model_revisions=dict(diffa=DIFFA_REVISION,llm=LLM_REVISION,whisper=WHISPER_REVISION),
        decoder=dict(answer_slots=512,block_length=32,max_forwards=256,stopping='eos',commit='fixed-quota-top-confidence'))
    args.out.mkdir(parents=True,exist_ok=True)
    config_path=args.out/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=json.loads(json.dumps(config)):
            raise ValueError('Source/config/inputs changed; use a fresh run name')
    elif any(args.out.iterdir()):
        raise ValueError('Output directory contains files without a config')
    else:
        write_json(config_path,config)
    print(f'Suite={args.suite}; exact previous {len(rows)} questions; baseline_mode={args.baseline_mode}; '
          f'conditions={[s["condition"] for s in specs]}; one decode each, one process warmup. '
          'No extra diagnostics or replay.',flush=True)
    if all((args.out/'samples'/f'{index:05d}'/'result.json').exists() for index in range(len(rows))):
        print(summarize(args.out),flush=True)
        print(f'Already complete: {args.out}/summary.json',flush=True)
        return
    model,tokenizer,extractor,load_report=load_model(args.diffa_root,DIFFA_REPO,DIFFA_REVISION,
        LLM_REPO,LLM_REVISION,'openai/whisper-large-v3',WHISPER_REVISION)
    llm=model.llm_model
    def execute(spec,completed):
        resolved=resolve_policy(spec,completed)
        intervals=resolved.get('intervals')
        if resolved.get('pruning') or resolved.get('sparse'):
            return generation(llm,prepared,tokenizer,args,intervals,policy=resolved)
        return generation(llm,prepared,tokenizer,args,intervals)
    write_json(args.out/'load_report.json',load_report)
    write_json(args.out/'device.json',dict(name=torch.cuda.get_device_name(),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory))
    from src.origin_generate import generate as upstream_generate
    samples=args.out/'samples'
    samples.mkdir(exist_ok=True)
    if args.reuse_run:
        if args.reuse_run.resolve()==args.out.resolve():
            raise ValueError('Use a new output directory when reusing an interrupted run')
        previous=json.loads((args.reuse_run/'config.json').read_text())
        if (previous['development50_sha256']!=config['development50_sha256']
                or previous['model_revisions']!=config['model_revisions'] or previous['decoder']!=config['decoder']
                or previous['settings']['seed']!=args.seed or previous['torch']!=config['torch']):
            raise ValueError('Previous run model/input/decoder/seed/environment mismatch')
        requested={'cache_a%d_p%d_s%d'%POLICIES[name] for name in args.policies}
        for index,row in enumerate(rows):
            target=samples/f'{index:05d}'
            if (target/'result.json').exists():
                continue
            source=args.reuse_run/'samples'/f'{index:05d}'/'result.json'
            if not source.exists():
                continue
            result=json.loads(source.read_text())
            if (result['id'],result['prompt'],result['target'])!=(row['id'],row['prompt'],row['target']):
                raise ValueError('Previous sample identity mismatch')
            policies=[p for p in result['policies'] if p['condition'] in requested]
            if {p['condition'] for p in policies}!=requested:
                continue
            if args.baseline_mode=='saved':
                result['baseline']=grade(saved['measurements'][row['id']],'baseline',row['target'])
            result['policies']=policies
            result['reused_from']=str(source.resolve())
            target.mkdir()
            write_json(target/'result.json',result)
            print(f'Reused completed cache result: {row["id"]}',flush=True)
    warmed=False
    for index,row in enumerate(rows):
        destination=samples/f'{index:05d}'
        if (destination/'result.json').exists():
            print(f'Skip completed {row["id"]}',flush=True)
            continue
        work=samples/f'.{index:05d}.incomplete'
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        print(f'[{index+1}/{len(rows)}] {row["id"]}',flush=True)
        audio,duration=load_audio(row['audio'])
        prepared=prepare(model,tokenizer,extractor,audio,row['prompt'],512)
        if not warmed:
            print('One process warmup: requested conditions'+(' and baseline.' if args.baseline_mode=='fresh' else ' only.'),flush=True)
            if args.baseline_mode=='fresh':
                generation(llm,prepared,tokenizer,args)
            completed={}
            for spec in specs:
                _,measurement=execute(spec,completed)
                completed[spec['condition']]=measurement
            warmed=True
        if args.baseline_mode=='fresh':
            final,baseline=generation(llm,prepared,tokenizer,args)
            del final
        else:
            baseline=saved['measurements'][row['id']]
        verified=None
        if index<args.verify_samples:
            official=upstream_generate(SimpleNamespace(llm_model=llm,device=llm.device),prepared.prompt,
                prepared.prompt.new_tensor([prepared.insertion]),prepared.speech,steps=256,gen_length=512,
                block_length=32,temperature=0.,cfg_scale=0.,mask_id=prepared.mask_id)
            ids=official[0,prepared.prompt.shape[1]:].tolist()
            visible=ids[:ids.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in ids else ids
            verified=visible==baseline['visible_answer_ids']
            if not verified:
                raise RuntimeError('Baseline differs from pinned official decoder')
            del official
        baseline=grade(baseline,'baseline',row['target'])
        print(f'  baseline ({args.baseline_mode}): correct={baseline["secondary_correct"]}; official_match={verified}',flush=True)
        result=dict(id=row['id'],prompt=row['prompt'],target=row['target'],cohort='development50',
            audio_seconds=duration,group_tokens={g:len(prepared.group_positions(g)) for g in ('acoustic','prompt','semantic')},
            verified_against_upstream=verified,baseline=baseline,policies=[])
        order=list(specs)
        if args.suite!='prune':
            random.Random(f'{args.seed}:{row["id"]}').shuffle(order)
        completed={}
        for spec in order:
            label=spec['condition']
            print(f'  {label}: starting',flush=True)
            final,measurement=execute(spec,completed)
            del final
            measurement=grade(measurement,label,row['target'])
            if spec.get('intervals'):
                measurement['intervals_acoustic_prompt_semantic']=spec['intervals']
            measurement['specification']=spec
            completed[label]=measurement
            result['policies'].append(measurement)
            write_json(work/'partial.json',result)
            print(f'    {measurement["seconds"]:.2f}s; {measurement["accounting"][0]["forwards"]} forwards; '
                  f'{measurement["visible_answer_tokens"]/measurement["seconds"]:.2f} TPS; '
                  f'correct={measurement["secondary_correct"]}',flush=True)
        write_json(work/'result.json',result)
        (work/'partial.json').unlink(missing_ok=True)
        work.rename(destination)
        summarize(args.out)
    print(summarize(args.out),flush=True)
    print(f'Complete: {args.out}/summary.json',flush=True)


if __name__=='__main__':
    main()
