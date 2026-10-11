"""Ten matched FLORES Chinese→English examples; TOP1 and threshold traces."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

from confidence_geography.run import MODEL, REVISION, collect_sample, command, dump, load_model, run_lock
from .evaluate import SCORING_PROTOCOL, evaluate, score_translation

HERE = Path(__file__).resolve().parent
PROMPT_PROTOCOL = 'flores_zh_en_translation_only_chat_v1'
INSTRUCTION = 'Translate the following Chinese text into English.\nReturn only the translation.\n\n'


def load_samples(path, count):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if count < 1 or count > len(rows):
        raise ValueError(f'Requested {count} examples; input has {len(rows)}')
    selected = rows[:count]
    if len({r['id'] for r in selected}) != count:
        raise ValueError('Duplicate sample IDs')
    for row in selected:
        if not isinstance(row['id'], str) or not row['id'].isascii() or not row['id'].isdigit():
            raise ValueError('Sample IDs must be numeric')
        if row.get('source_language') != 'zho_Hans' or row.get('target_language') != 'eng_Latn':
            raise ValueError('Expected Simplified Chinese→English inputs')
        if not row['source_text'].strip() or not row['reference'].strip():
            raise ValueError('Empty source/reference')
    return [{**r, 'question': INSTRUCTION+r['source_text'], 'answer': r['reference']} for r in selected]


def fingerprint():
    paths = [*HERE.glob('*.py'), Path(__file__).resolve().parents[2]/'confidence_geography/run.py',
             Path(__file__).resolve().parents[2]/'confidence_geography/core.py']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(paths))).hexdigest()


def run(args):
    import numpy as np
    import torch
    if args.length < 1 or args.log_every < 1 or not 0 <= args.commit_threshold <= 1:
        raise ValueError('Positive length/log interval and threshold in [0,1] required')
    samples = load_samples(args.inputs, args.samples)
    config = {'model': MODEL, 'revision': REVISION, 'mask_id': 126336, 'length': args.length,
              'block_length': args.length, 'policies': args.policies, 'seed': args.seed,
              'samples': args.samples, 'commit_threshold': args.commit_threshold,
              'thresholds': [.5,.7,.8,.9,.95,.99], 'top_k': 5, 'log_every': args.log_every,
              'prompt_protocol': PROMPT_PROTOCOL, 'scoring_protocol': SCORING_PROTOCOL,
              'inputs_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest()}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        manifest_path = out/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['config'] != config or manifest['source_sha256'] != fingerprint():
                raise ValueError('Run arguments/source changed; choose a new --out')
            stored = [json.loads(line) for line in (out/'samples.jsonl').read_text().splitlines()]
            if stored != samples:
                raise ValueError('Saved inputs differ from requested inputs')
        else:
            if any(out.glob('*/samples/*/result.json')):
                raise ValueError('Existing results without a manifest; choose a new --out')
            (out/'samples.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in samples),encoding='utf-8')
            manifest = {'config': config, 'source_sha256': fingerprint(), 'created_unix': time.time(),
                        'git_commit': command('git','rev-parse','HEAD'),
                        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                        'dataset_info': json.loads((args.inputs.parent/'provenance.json').read_text()) if (args.inputs.parent/'provenance.json').exists() else {'source':'custom_jsonl'},
                        'instruction': INSTRUCTION, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
            dump(manifest_path,manifest)
        pending=[]
        for policy in args.policies:
            for sample in samples:
                directory=out/policy/'samples'/sample['id']
                if (directory/'result.json').exists():
                    if not (directory/'trace.jsonl.gz').exists():
                        raise ValueError(f'Completed result missing trace: {directory}')
                    continue
                pending.append((policy,sample,directory))
        if pending:
            print(f'Translation run: {len(samples)} matched examples; policies={args.policies}; pending={len(pending)}; length={args.length}',flush=True)
            print('Loading pinned LLaDA checkpoint...',flush=True)
            model,tokenizer=load_model(config)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            for policy,sample,directory in pending:
                # Reset per sample so resuming cannot alter an RNG-dependent path.
                random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
                directory.mkdir(parents=True,exist_ok=True);torch.cuda.reset_peak_memory_stats()
                result=collect_sample(model,tokenizer,{**config,'policy':policy},sample,directory,
                                      prompt_protocol=PROMPT_PROTOCOL,score_output=score_translation)
                print(f"DONE policy={policy} sample={sample['id']} chrF++={result['chrfpp']:.2f} forwards={result['steps']} seconds={result['elapsed_seconds']:.1f}",flush=True)
            del model
        evaluate(out)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=HERE/'inputs/flores_zh_en_dev_10.jsonl')
    p.add_argument('--samples',type=int,default=10)
    p.add_argument('--policies',nargs='+',choices=['top1','threshold'],default=['top1','threshold'])
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--commit-threshold',type=float,default=.9)
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--log-every',type=int,default=32)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if len(set(args.policies)) != len(args.policies):raise ValueError('Duplicate policies')
    run(args)
