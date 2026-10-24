"""Dedicated cache, global-pruning and audio-sparse experiments. No drift/boost paths."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import torch

from experiments.audio_drift.checkout import DEFAULT_ROOT, COMMIT, verify_checkout
from experiments.audio_drift.compact import PrefixCache
from experiments.audio_drift.decode import decode
from experiments.audio_drift.inputs import read_manifest, load_audio, score, canonical_number
from experiments.audio_drift.model import (load_model, prepare, DIFFA_REPO, DIFFA_REVISION,
                                          LLM_REPO, LLM_REVISION)
from experiments.audio_drift.numeric_answer_report import extract_answer, VERSION
from experiments.audio_drift.progressive_pruning import progressive_conditions, resolve_progressive_control
from .pruning import GlobalPruning
from .sparse import AudioSparseAttention

DEV50_SHA = '8ed78386500a79daecdf9b9be573efe2c2a4001a5c667b23c58555a6e03205f2'
WHISPER_REVISION = '06f233fe06e710322aca913c1bc4249a0d71fce1'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def fingerprint(rows):
    content = [{k: r[k] for k in ('id','source_revision','prompt','target','metric','audio_sha256')} for r in rows]
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def cache_spec(a, p, s):
    return dict(condition=f'cache_a{a}_p{p}_s{s}', schedule=[dict(group=g, refresh_every=r)
        for g,r in [('acoustic',a),('prompt',p),('semantic',s)]])


def specifications(args):
    if args.suite == 'cache':
        return [cache_spec(8,8,8), cache_spec(16,8,16), cache_spec(16,16,16)]
    if args.suite == 'sparse':
        sparse = dict(chunk_size=args.chunk_size, dense_layers=args.dense_layers, layer_group=8,
                      anchor_fractions=args.anchor_fractions, proxy_queries=args.proxy_queries, neighbors=1)
        control = cache_spec(8,8,8)
        return [control, dict(condition='sparse_audio', sparse=sparse),
                dict(**{**control, 'condition':'sparse_audio_cache888'}, sparse=sparse)]
    result = []
    scopes = [args.query_scope]
    if args.compare_current_block and args.query_scope != 'current_block':
        scopes.append('current_block')
    for scope in scopes:
        pair = [s for s in progressive_conditions(fraction=args.prune_fraction, threshold=args.prune_threshold,
                layer_start=args.prune_layer_start) if s['pruning']['control'] in ('audio','generic')]
        for spec in pair:
            spec['condition'] = scope+'__'+spec['condition']
            for field in ('control_source','paired_condition'):
                if field in spec:
                    spec[field] = scope+'__'+spec[field]
            spec['pruning'].update(query_scope=scope, query_chunk=args.query_chunk)
            result.append(spec)
    return result


def ordered(specs, sample_id, seed):
    result = deepcopy(specs)
    random.Random(f'{seed}:{sample_id}').shuffle(result)
    done, output = set(), []
    while result:
        ready = next(s for s in result if not s.get('control_source') or s['control_source'] in done)
        result.remove(ready)
        output.append(ready)
        done.add(ready['condition'])
    return output


@torch.no_grad()
def generation(llm, prepared, tokenizer, args, spec=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    engine, sparse = None, None
    with ExitStack() as stack:
        if spec:
            if spec.get('pruning'):
                engine = stack.enter_context(GlobalPruning(llm, prepared, deepcopy(spec['pruning'])))
            elif spec.get('schedule'):
                engine = stack.enter_context(PrefixCache(llm, prepared, spec['schedule']))
            if spec.get('sparse'):
                sparse = stack.enter_context(AudioSparseAttention(llm, prepared, 512, cache=engine, **spec['sparse']))
        final, _, events = decode(llm, prepared, 512, 32, 256, save_states=False,
                                 eos_id=tokenizer.eos_token_id, progress_every=args.progress_every)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    # Accounting/scoring/output extraction are deliberately outside decode timing.
    account = engine.accounting() if engine else dict(forwards=len(events))
    if sparse:
        account['sparse'] = sparse.accounting()
    if spec and spec.get('pruning'):
        pruning = account['pruning']
        pruning['trigger_checks'] = [{k:r[k] for k in ('step','apply_step','current_key_count',
            'eligible_by_group','planned_removals')} for r in pruning['trigger_checks']]
        for selection in pruning['selections']:
            selection.pop('probe_positions', None)
    ids = final[0, prepared.prompt.shape[1]:].tolist()
    eos = tokenizer.eos_token_id
    visible = ids[:ids.index(eos)] if eos in ids else ids
    return final, dict(seconds=seconds, visible_answer_tokens=len(visible), visible_answer_ids=visible,
                      answer_has_no_eos=eos not in ids, peak_allocated_bytes=peak,
                      text=tokenizer.decode(visible, skip_special_tokens=True).strip(), accounting=[account])


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
        'No drift recording or replay. Sparse/pruning observation overhead is included in timing.', '',
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
        summary[name] = dict(samples=n, correct=correct, accuracy=correct/n, accuracy_delta_pp=delta,
            tps=tps, tps_change_percent=100*(tps/base_tps-1), decode_speedup=base_seconds/seconds,
            wins=wins, losses=losses, parse_failures=sum(r['parse_failed'] for r in selected))
        lines.append(f'| {name} | {n} | {correct} | {100*correct/n:.1f}% | {delta:+.1f} | '
                     f'{tps:.2f} | {100*(tps/base_tps-1):+.1f}% | {base_seconds/seconds:.3f}× | {wins} / {losses} |')
    analysis = out/'analysis'
    analysis.mkdir(exist_ok=True)
    (analysis/'report.md').write_text('\n'.join(lines)+'\n')
    with (analysis/'pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out/'summary.json', summary)
    return '\n'.join(lines)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['cache','prune','sparse'], required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--diffa-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--query-scope', choices=['all','all_masked','current_block'], default='all')
    p.add_argument('--compare-current-block', action='store_true')
    p.add_argument('--query-chunk', type=int, default=32)
    p.add_argument('--prune-fraction', type=float, default=.125)
    p.add_argument('--prune-threshold', type=float, default=.5)
    p.add_argument('--prune-layer-start', type=int, choices=[0,25], default=0)
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--dense-layers', type=int, default=8)
    p.add_argument('--anchor-fractions', type=float, nargs=3, default=[.5,.25,.125])
    p.add_argument('--proxy-queries', type=int, default=128)
    p.add_argument('--verify-samples', type=int, default=2)
    p.add_argument('--progress-every', type=int, default=32)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=1729)
    return p


@torch.no_grad()
def main():
    args = parser().parse_args()
    if not 1 <= args.samples <= 50 or min(args.cpu_threads,args.query_chunk,args.chunk_size,args.proxy_queries)<1:
        raise ValueError('Use 1–50 development questions and positive chunk/thread settings')
    if args.verify_samples < 0 or args.progress_every < 0:
        raise ValueError('Invalid verification/progress setting')
    if args.suite == 'sparse' and args.dense_layers != 8:
        raise ValueError('This initial sparse experiment uses dense layers0–7, then three groups of8')
    if args.suite == 'sparse' and any(not math.isfinite(f) or not 0 < f <= 1 for f in args.anchor_fractions):
        raise ValueError('Anchor fractions must be finite and in (0,1]')
    specs = specifications(args)
    verify_checkout(args.diffa_root)
    full = read_manifest(args.manifest,50)
    if len(full)!=50 or fingerprint(full)!=DEV50_SHA:
        raise ValueError('Manifest is not the exact earlier development50 cohort')
    rows = full[:args.samples]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run on a BF16 CUDA GPU')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = hashlib.sha256()
    for directory in (Path(__file__).parent, Path(__file__).parent.parent/'audio_drift'):
        for path in sorted(directory.glob('*.py')):
            source.update(str(path.relative_to(directory.parent)).encode())
            source.update(path.read_bytes())
    config = dict(protocol='audio-clean-v27', settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        specifications=specs, inputs=rows, development50_sha256=DEV50_SHA, source_sha256=source.hexdigest(),
        upstream_commit=COMMIT, torch=torch.__version__, scorer=VERSION,
        model_revisions=dict(diffa=DIFFA_REVISION, llm=LLM_REVISION, whisper=WHISPER_REVISION),
        decoder=dict(answer_slots=512,block_length=32,max_forwards=256,stopping='eos',commit='fixed-quota-top-confidence'))
    args.out.mkdir(parents=True,exist_ok=True)
    config_path = args.out/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Source/config/input changed; use a fresh RUN_NAME')
    elif any(args.out.iterdir()):
        raise ValueError('Output directory has files but no configuration')
    else:
        write_json(config_path,config)
    print(f'{args.suite}: {len(rows)} exact development questions; baseline + {len(specs)} policies; '
          'one decode each, one process warmup; no drift, boost, profiling or replay.',flush=True)
    model, tokenizer, extractor, load_report = load_model(args.diffa_root, DIFFA_REPO, DIFFA_REVISION,
        LLM_REPO, LLM_REVISION, 'openai/whisper-large-v3', WHISPER_REVISION)
    write_json(args.out/'load_report.json',load_report)
    write_json(args.out/'device.json',dict(name=torch.cuda.get_device_name(),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory))
    llm = model.llm_model
    from src.origin_generate import generate as upstream_generate
    samples = args.out/'samples'
    samples.mkdir(exist_ok=True)
    warmed = False
    for index,row in enumerate(rows):
        destination = samples/f'{index:05d}'
        if (destination/'result.json').exists():
            print(f'Skip complete {row["id"]}',flush=True)
            continue
        work = samples/f'.{index:05d}.incomplete'
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        print(f'[{index+1}/{len(rows)}] {row["id"]}',flush=True)
        audio, duration = load_audio(row['audio'])
        prepared = prepare(model,tokenizer,extractor,audio,row['prompt'],512)
        if not warmed:
            print('Process warmup: baseline and each implementation, once.',flush=True)
            _, _ = generation(llm,prepared,tokenizer,args)
            completed = {}
            for spec in specs:
                resolved = resolve_progressive_control(spec,completed[spec['control_source']]) if spec.get('control_source') else spec
                _, measurement = generation(llm,prepared,tokenizer,args,resolved)
                completed[spec['condition']] = measurement
            warmed = True
        final, baseline = generation(llm,prepared,tokenizer,args)
        verified = None
        if index < args.verify_samples:
            official = upstream_generate(SimpleNamespace(llm_model=llm,device=llm.device), prepared.prompt,
                prepared.prompt.new_tensor([prepared.insertion]), prepared.speech, steps=256,gen_length=512,
                block_length=32,temperature=0.,cfg_scale=0.,mask_id=prepared.mask_id)
            ids = official[0,prepared.prompt.shape[1]:].tolist()
            visible = ids[:ids.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in ids else ids
            verified = visible == baseline['visible_answer_ids']
            if not verified:
                raise RuntimeError('Baseline differs from pinned official decoder')
            del official
        del final

        def grade(measurement, name):
            prediction, method = extract_answer(measurement['text'])
            return dict(**measurement, condition=name, score=score(measurement['text'],row['target'],'numeric'),
                secondary_prediction=prediction, secondary_correct=prediction==canonical_number(row['target']),
                secondary_parse_failed=prediction is None, extraction=method)

        baseline = grade(baseline,'baseline')
        print(f'  baseline: {baseline["seconds"]:.2f}s; {baseline["accounting"][0]["forwards"]} forwards; '
              f'correct={baseline["secondary_correct"]}; official_match={verified}',flush=True)
        result = dict(id=row['id'],prompt=row['prompt'],target=row['target'],cohort='development50_v27',
            audio_seconds=duration,group_tokens={g:len(prepared.group_positions(g)) for g in ('acoustic','semantic','prompt')},
            verified_against_upstream=verified,baseline=baseline,policies=[])
        completed = {}
        for spec in ordered(specs,row['id'],args.seed):
            resolved = resolve_progressive_control(spec,completed[spec['control_source']]) if spec.get('control_source') else spec
            print(f'  {spec["condition"]}: starting',flush=True)
            final, measurement = generation(llm,prepared,tokenizer,args,resolved)
            del final
            measurement = grade(measurement,spec['condition'])
            measurement['specification'] = resolved
            completed[spec['condition']] = measurement
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


if __name__ == '__main__':
    main()
