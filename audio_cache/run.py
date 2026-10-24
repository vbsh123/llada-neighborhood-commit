"""Baseline and fixed-prefix cache schedules on the original50 spoken questions."""
import argparse
from contextlib import nullcontext
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
POLICIES={'888':(8,8,8),'16816':(16,8,16),'161616':(16,16,16)}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/development50.jsonl'))
    p.add_argument('--out',type=Path,default=Path('runs/cache161616_50'))
    p.add_argument('--policies',choices=POLICIES,nargs='+',default=['161616'])
    p.add_argument('--samples',type=int,default=50)
    p.add_argument('--diffa-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--verify-samples',type=int,default=2)
    p.add_argument('--progress-every',type=int,default=32)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=1729)
    return p


@torch.no_grad()
def generation(llm,prepared,tokenizer,args,intervals=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    context=PrefixCache(llm,prepared,intervals) if intervals else nullcontext()
    with context as cache:
        final,events=decode(llm,prepared,512,32,256,
            eos_id=tokenizer.eos_token_id,progress_every=args.progress_every)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    account=cache.accounting() if intervals else dict(forwards=len(events))
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
    verify_checkout(args.diffa_root)
    all_rows=read_manifest(args.manifest)
    rows=all_rows[:args.samples]
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
    config=dict(protocol='standalone-audio-cache-v28',
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        inputs=rows,development50_sha256=fingerprint(all_rows),source_sha256=source.hexdigest(),
        upstream_commit=COMMIT,torch=torch.__version__,scorer=VERSION,
        model_revisions=dict(diffa=DIFFA_REVISION,llm=LLM_REVISION,whisper=WHISPER_REVISION),
        decoder=dict(answer_slots=512,block_length=32,max_forwards=256,stopping='eos',commit='fixed-quota-top-confidence'))
    args.out.mkdir(parents=True,exist_ok=True)
    config_path=args.out/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:
            raise ValueError('Source/config/inputs changed; use a fresh run name')
    elif any(args.out.iterdir()):
        raise ValueError('Output directory contains files without a config')
    else:
        write_json(config_path,config)
    print(f'Exact previous {len(rows)} questions; baseline + {args.policies}; one decode each, '
          'one process warmup. No pruning, sparse attention, drift, boosting or replay.',flush=True)
    model,tokenizer,extractor,load_report=load_model(args.diffa_root,DIFFA_REPO,DIFFA_REVISION,
        LLM_REPO,LLM_REVISION,'openai/whisper-large-v3',WHISPER_REVISION)
    llm=model.llm_model
    write_json(args.out/'load_report.json',load_report)
    write_json(args.out/'device.json',dict(name=torch.cuda.get_device_name(),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory))
    from src.origin_generate import generate as upstream_generate
    samples=args.out/'samples'
    samples.mkdir(exist_ok=True)
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
            print('One process warmup: baseline and requested cache policies.',flush=True)
            generation(llm,prepared,tokenizer,args)
            for name in args.policies:
                generation(llm,prepared,tokenizer,args,POLICIES[name])
            warmed=True
        final,baseline=generation(llm,prepared,tokenizer,args)
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
        del final
        baseline=grade(baseline,'baseline',row['target'])
        print(f'  baseline: {baseline["seconds"]:.2f}s; {baseline["accounting"][0]["forwards"]} forwards; '
              f'correct={baseline["secondary_correct"]}; official_match={verified}',flush=True)
        result=dict(id=row['id'],prompt=row['prompt'],target=row['target'],cohort='development50',
            audio_seconds=duration,group_tokens={g:len(prepared.group_positions(g)) for g in ('acoustic','prompt','semantic')},
            verified_against_upstream=verified,baseline=baseline,policies=[])
        order=list(args.policies)
        random.Random(f'{args.seed}:{row["id"]}').shuffle(order)
        for name in order:
            intervals=POLICIES[name]
            label='cache_a%d_p%d_s%d'%intervals
            print(f'  {label}: starting',flush=True)
            final,measurement=generation(llm,prepared,tokenizer,args,intervals)
            del final
            measurement=grade(measurement,label,row['target'])
            measurement['intervals_acoustic_prompt_semantic']=intervals
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
