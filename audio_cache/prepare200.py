"""Freeze200 cache-comparison questions, with explicit fresh/reused provenance."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random

import soundfile as sf

from .data import fingerprint,prepare_manifest
from .model import DIFFA_REVISION,LLM_REVISION
from .scoring import canonical_number
from .storage import write_json

DATASET='mistralai/gsm8k_speech'
REVISION='e94caa2154433a54cd430702666c454977df3083'
WHISPER_REVISION='06f233fe06e710322aca913c1bc4249a0d71fce1'
PROMPT=('Solve the math problem spoken in the audio. Think step by step, then end your response '
        'with a separate line: #### <final numerical answer>. '
        'Write only the number after ####, without units or additional text.')


def reservoir(rows,count,seed):
    rng=random.Random(seed)
    selected=[]
    for seen,row in enumerate(rows,1):
        slot=len(selected) if len(selected)<count else rng.randrange(seen)
        if slot<count:
            if slot==len(selected):selected.append(row)
            else:selected[slot]=row
    if len(selected)!=count:
        raise ValueError(f'Only{len(selected)} eligible unused questions; need{count}')
    return selected


def source_rows(files):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    index=0
    for name in files:
        print(f'Scanning pinned dataset shard: {name}',flush=True)
        local=hf_hub_download(DATASET,name,repo_type='dataset',revision=REVISION)
        with open(local,'rb') as handle,pq.ParquetFile(handle) as parquet:
            offset=0
            for batch in parquet.iter_batches(batch_size=4,use_threads=False):
                for row in batch.to_pylist():
                    yield index,name,offset,row
                    index+=1
                    offset+=1


def fresh_reference(seed):
    from huggingface_hub import HfApi
    earlier=json.loads(Path(__file__).with_name('earlier130_ids.json').read_text())
    if earlier['source_revision']!=REVISION or earlier['source_dataset']!=DATASET:
        raise ValueError('Earlier exclusion revision mismatch')
    excluded=set(earlier['ids'])
    files=sorted(f for f in HfApi().list_repo_files(DATASET,repo_type='dataset',revision=REVISION)
                 if f.startswith('data/test-') and f.endswith('.parquet'))
    if not files:raise ValueError('Pinned test shards missing')
    eligible=[]
    for index,name,offset,row in source_rows(files):
        identifier=f'gsm8k-speech-{index:05d}'
        if identifier in excluded:continue
        info=sf.info(io.BytesIO(row['audio']['bytes']))
        seconds=info.frames/info.samplerate
        if not 0<=seconds<=30:continue
        target=canonical_number(str(row['answer']))
        if target is None:raise ValueError('Invalid numeric source answer')
        eligible.append(dict(id=identifier,source_file=name,source_row=offset,source_index=index,
            audio_seconds=seconds,target=target))
    # Exactly reconstruct the old v23/v25 selection: same ordered eligible rows,
    # reservoir algorithm, sample count and seed. No old model outputs are used.
    previous500=reservoir(eligible,500,20261024)
    excluded.update(r['id'] for r in previous500)
    chosen=reservoir([r for r in eligible if r['id'] not in excluded],200,seed)
    by_source={(r['source_file'],r['source_row']):r for r in chosen}
    hashes={}
    for _,name,offset,row in source_rows(files):
        selected=by_source.get((name,offset))
        if selected is None:continue
        audio,rate=sf.read(io.BytesIO(row['audio']['bytes']),dtype='float32')
        exported=io.BytesIO()
        sf.write(exported,audio,rate,format='WAV',subtype='PCM_16')
        hashes[selected['id']]=hashlib.sha256(exported.getvalue()).hexdigest()
    rows=[]
    for r in chosen:
        rows.append({**r,'prompt':PROMPT,'metric':'numeric','task':'gsm8k_speech',
            'cohort':'cache_fresh200_v31','source_dataset':DATASET,'source_revision':REVISION,
            'audio_sha256':hashes[r['id']]})
    return dict(protocol='cache-comparison-cohort-v31',mode='fresh',seed=seed,rows=rows,
        sha256=fingerprint(rows),source_revision=REVISION,source_files=files,
        excluded_ids=sorted(excluded),eligible_after_earlier130=len(eligible),
        previous500_selection=dict(samples=500,seed=20261024,algorithm='uniform reservoir;0–30seconds;exclude earlier130'),
        selection='Uniform reservoir over remaining duration-eligible rows; no correctness filtering')


def verify_baseline_config(config):
    expected=dict(diffa=DIFFA_REVISION,llm=LLM_REVISION,whisper=WHISPER_REVISION)
    if config.get('resolved_revisions',config.get('model_revisions'))!=expected:
        raise ValueError('Reference run uses different model revisions')
    settings=config['settings']
    decoder=config.get('decoder',dict(answer_slots=settings.get('length'),block_length=settings.get('block_length'),
        max_forwards=settings.get('steps'),stopping='eos',commit='fixed-quota-top-confidence'))
    if decoder!=dict(answer_slots=512,block_length=32,max_forwards=256,stopping='eos',commit='fixed-quota-top-confidence'):
        raise ValueError('Reference run decoder differs')
    if settings.get('seed')!=1729 or settings.get('lora_alpha',16)!=16:
        raise ValueError('Reference seed/LoRA scaling differs')


def find_baseline_run(roots=None,seed=20261025):
    roots=roots or [Path.cwd(),Path.cwd().parent]
    configs=set()
    for root in map(Path,roots):
        if (root/'config.json').is_file():configs.add(root/'config.json')
        for pattern in ('runs/*/config.json','*/runs/*/config.json','*/config.json'):
            configs.update(root.glob(pattern))
    for path in sorted(configs,key=str):
        try:
            config=json.loads(path.read_text())
            verify_baseline_config(config)
            rows=config['inputs']
            if len(rows)<200 or len({r['id'] for r in rows})!=len(rows):continue
            indices=random.Random(seed).sample(range(len(rows)),200)
            for index in indices:
                if (rows[index].get('source_dataset')!=DATASET or rows[index].get('source_revision')!=REVISION
                        or rows[index].get('metric')!='numeric'):
                    raise ValueError('Reference run is not pinned spoken GSM')
                result=json.loads((path.parent/'samples'/f'{index:05d}'/'result.json').read_text())
                if (result['id'],result['prompt'],result['target'])!=(rows[index]['id'],rows[index]['prompt'],rows[index]['target']):
                    raise ValueError('Baseline sample identity mismatch')
                if 'baseline' not in result and 'baseline_median_seconds' not in result.get('benchmark',{}):
                    raise ValueError('Missing baseline measurement')
            return path.parent.resolve()
        except (OSError,ValueError,KeyError,TypeError):
            continue
    raise ValueError('No compatible saved run with200 complete baseline results found. '
                     'Set BASELINE_RUN to an extracted run containing config.json and samples/*/result.json. '
                     'No new baseline inference was started.')


def reused_reference(run,seed):
    run=Path(run).resolve()
    config=json.loads((run/'config.json').read_text())
    verify_baseline_config(config)
    inputs=config['inputs']
    if len({r['id'] for r in inputs})!=len(inputs) or len(inputs)<200:
        raise ValueError('Need at least200 unique reference questions')
    indices=random.Random(seed).sample(range(len(inputs)),200)
    rows=[]
    measurements={}
    for index in indices:
        row=dict(inputs[index])
        if (row.get('source_dataset')!=DATASET or row.get('source_revision')!=REVISION or row.get('metric')!='numeric'):
            raise ValueError('Reference questions are not the pinned spoken GSM data')
        record=json.loads((run/'samples'/f'{index:05d}'/'result.json').read_text())
        if (record['id'],record['prompt'],record['target'])!=(row['id'],row['prompt'],row['target']):
            raise ValueError('Reference sample identity mismatch')
        if 'baseline' in record:
            original=record['baseline']
            allowed=('seconds','visible_answer_tokens','visible_answer_ids','answer_has_no_eos','text','accounting')
            measurement={k:original[k] for k in allowed}
        else:
            benchmark=record['benchmark']
            ids=record['answer_ids']
            stop=ids.index(126081) if 126081 in ids else None
            visible=ids[:stop] if stop is not None else ids
            if len(visible)!=benchmark['baseline_visible_answer_tokens']:
                raise ValueError('Reference visible token count differs')
            measurement=dict(seconds=benchmark['baseline_median_seconds'],visible_answer_tokens=len(visible),
                visible_answer_ids=visible,answer_has_no_eos=stop is None,text=record['text'],
                accounting=[dict(forwards=benchmark['baseline_forwards'])])
        measurement.update(historical_timing=True,reference_run=str(run))
        measurements[row['id']]=measurement
        if row.get('audio'):row['original_audio']=row['audio']
        row.pop('audio',None)
        row['cohort']='cache_reused200_v31'
        rows.append(row)
    reference=dict(protocol='cache-comparison-cohort-v31',mode='reused',seed=seed,rows=rows,
        sha256=fingerprint(rows),source_revision=REVISION,source_run=str(run),
        source_config_sha256=hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),source_indices=indices,
        selection='Seeded sample of200 reference questions; no correctness filtering; previously evaluated data')
    saved=dict(input_sha256=reference['sha256'],measurements=measurements,reference_run=str(run))
    return reference,saved


def prepare(out,seed=20261025,baseline_run=None,reuse_only=False):
    out=Path(out)
    reference_path=out.with_suffix('.cohort.json')
    saved_path=out.with_suffix('.baselines.json')
    mode='reused' if baseline_run or reuse_only else 'fresh'
    if mode=='reused' and not baseline_run and not reference_path.exists():
        baseline_run=find_baseline_run(seed=seed)
        print(f'Using existing baseline run: {baseline_run}',flush=True)
    if reference_path.exists():
        reference=json.loads(reference_path.read_text())
        if reference['mode']!=mode or reference['seed']!=seed or len(reference['rows'])!=200:
            raise ValueError('Prepared cohort settings changed; use a fresh manifest')
        if baseline_run and reference['source_run']!=str(Path(baseline_run).resolve()):
            raise ValueError('Reference baseline run changed')
        if mode=='reused' and not saved_path.exists():
            raise ValueError('Missing saved baseline sidecar')
    else:
        if out.exists():raise ValueError('Existing manifest has no frozen cohort sidecar')
        out.parent.mkdir(parents=True,exist_ok=True)
        if baseline_run:
            reference,saved=reused_reference(baseline_run,seed)
            write_json(saved_path,saved)
        else:
            reference=fresh_reference(seed)
        write_json(reference_path,reference)
    prepare_manifest(out,reference_path)
    print(f'Prepared200: mode={mode}; input_sha256={reference["sha256"]}; reference={reference_path}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--seed',type=int,default=20261025)
    parser.add_argument('--baseline-run',type=Path)
    parser.add_argument('--reuse-only',action='store_true',help='Find/reuse existing baselines; never silently generate fresh ones')
    parser.add_argument('--find-baseline',action='store_true')
    parser.add_argument('--search-roots',type=Path,nargs='+')
    args=parser.parse_args()
    try:
        if args.find_baseline:
            print(find_baseline_run(args.search_roots,args.seed))
        else:
            if args.out is None:parser.error('--out is required')
            prepare(args.out,args.seed,args.baseline_run,args.reuse_only)
    except ValueError as error:
        parser.exit(2,f'Preparation stopped: {error}\n')
