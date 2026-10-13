"""Ten matched FLORES Chinese→English examples; TOP1 and threshold traces."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

from confidence_geography.run import MODEL, REVISION, collect_sample, command, dump, load_model, run_lock
from confidence_geography.core import select
from .evaluate import SCORING_PROTOCOL, evaluate, score_translation

HERE = Path(__file__).resolve().parent
PROMPT_PROTOCOL = 'flores_zh_en_translation_only_chat_v1'
INSTRUCTION = 'Translate the following Chinese text into English.\nReturn only the translation.\n\n'
ALL_INSTRUCTION = 'Translate all four Chinese sentences below into English, preserving their order.\nTranslate every sentence; do not omit any. Return only the translations.\n\n'
ALL_PROMPT_PROTOCOL = 'flores_zh_en_translate_all_four_chat_v1'


def select_translation(rows, config, stop_ids):
    """Full-window control or explicitly labeled stop-deferral diagnostic.

    Deferral never changes logits, raw probabilities, or predicted token IDs.
    While any eligible position predicts a non-stop token, stop predictions are
    withheld from commitment. If all remaining predictions are stop tokens,
    the ordinary selection rule applies. There is no spatial restriction.
    """
    mode = config.get('stop_policy', 'unchanged')
    if mode not in ('unchanged', 'defer'):
        raise ValueError('Unknown translation stop policy')
    candidates = [r for r in rows if r['eligible']]
    non_stop = [r for r in candidates if r['token_id'] not in stop_ids]
    deferred = [r['position'] for r in candidates if r['token_id'] in stop_ids] if mode == 'defer' and non_stop else []
    chosen = select(non_stop if deferred else rows, config['policy'], 1, config['commit_threshold'])
    return chosen, {'stop_selection': {'policy': mode, 'deferred_positions': deferred}}


def load_samples(path, count, instruction=INSTRUCTION):
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
    return [{**r, 'question': instruction+r['source_text'], 'answer': r['reference']} for r in selected]


def fingerprint():
    paths = [*HERE.glob('*.py'), Path(__file__).resolve().parents[2]/'confidence_geography/run.py',
             Path(__file__).resolve().parents[2]/'confidence_geography/core.py']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(paths))).hexdigest()


def run(args):
    if args.length < 1 or args.log_every < 1 or not 0 <= args.commit_threshold <= 1:
        raise ValueError('Positive length/log interval and threshold in [0,1] required')
    block_length = args.length if args.block_length is None else args.block_length
    if block_length != args.length:
        raise ValueError('Translation geography requires full-window decoding: block-length must equal length. Smaller blocks impose left-to-right progression.')
    import numpy as np
    import torch
    instruction = ALL_INSTRUCTION if args.translate_all else INSTRUCTION
    protocol = ALL_PROMPT_PROTOCOL if args.translate_all else PROMPT_PROTOCOL
    samples = load_samples(args.inputs, args.samples, instruction)
    if args.translate_all and any(len(s.get('sentence_pairs',[]))!=4 for s in samples):
        raise ValueError('--translate-all requires four-sentence inputs')
    config = {'model': MODEL, 'revision': REVISION, 'mask_id': 126336, 'length': args.length,
              'block_length': block_length, 'policies': args.policies, 'seed': args.seed,
              'stop_policy': args.stop_policy,
              'samples': args.samples, 'commit_threshold': args.commit_threshold,
              'thresholds': [.5,.7,.8,.9,.95,.99], 'top_k': 5, 'log_every': args.log_every,
              'prompt_protocol': protocol, 'scoring_protocol': SCORING_PROTOCOL,
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
                        'instruction': instruction, 'cuda': torch.version.cuda,
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
            print(f'Translation run: {len(samples)} matched examples; policies={args.policies}; pending={len(pending)}; length={args.length}; block_length={block_length}',flush=True)
            print(f'Selection: full-window; stop_policy={args.stop_policy}. No sequential blocks.',flush=True)
            if args.stop_policy == 'defer':
                print('DIAGNOSTIC: defer stop predictions while any eligible non-stop prediction remains; raw probabilities unchanged. This differs from the unchanged GSM8K decoder.',flush=True)
            print('Loading pinned LLaDA checkpoint...',flush=True)
            model,tokenizer=load_model(config)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            for policy,sample,directory in pending:
                # Reset per sample so resuming cannot alter an RNG-dependent path.
                random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
                directory.mkdir(parents=True,exist_ok=True);torch.cuda.reset_peak_memory_stats()
                result=collect_sample(model,tokenizer,{**config,'policy':policy},sample,directory,
                                      prompt_protocol=protocol,score_output=score_translation,
                                      select_output=select_translation)
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
    p.add_argument('--block-length',type=int,default=None,help='Compatibility option: must equal length. Smaller blocks are rejected.')
    p.add_argument('--stop-policy',choices=['unchanged','defer'],default='unchanged',
                   help='unchanged = original GSM8K selection; defer = explicit diagnostic withholding stop commits while non-stop predictions remain')
    p.add_argument('--commit-threshold',type=float,default=.9)
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--log-every',type=int,default=32)
    p.add_argument('--translate-all',action='store_true',help='Explicitly require translating all four sentences; diagnostic prompt variant')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if len(set(args.policies)) != len(args.policies):raise ValueError('Duplicate policies')
    run(args)
