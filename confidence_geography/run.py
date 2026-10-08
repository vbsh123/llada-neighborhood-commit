"""GSM8K collection. No KV cache, no sampling, no EOS suppression, no early stop."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import time

from .core import finite_json, nearest, numeric_answer, select, step_metrics, token_category

MODEL = 'GSAI-ML/LLaDA-8B-Instruct'
REVISION = '08b83a6feb34df1a6011b80c3c00c7563e963b07'
PROMPT_PROTOCOL = 'question_only_chat_v1'
SCORING_PROTOCOL = 'marked_answer_else_last_number_v1'


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(finite_json(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def emit(handle, obj):
    handle.write(json.dumps(finite_json(obj), ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


def command(*args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@contextmanager
def run_lock(out):
    import fcntl
    with (out / '.lock').open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process owns {out}') from exc
        yield


def load_model(config):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('Collection requires a CUDA GPU. Use the demo for CPU validation.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Collection requires BF16 GPU support.')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], trust_remote_code=True)
    if tokenizer.mask_token_id is not None and tokenizer.mask_token_id != config['mask_id']:
        raise ValueError('Configured mask ID does not match checkpoint tokenizer')
    model = AutoModel.from_pretrained(config['model'], revision=config['revision'],
                                     trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda').eval()
    return model, tokenizer


def token_info(tokenizer, token_id, special_ids):
    text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {'text': text, 'piece': tokenizer.convert_ids_to_tokens(token_id),
            'special': token_id in special_ids,
            'category': token_category(text, token_id in special_ids)}


def score_logits(logits, positions, previous, mask_id, top_k, chunk_size=16):
    """Exact vocabulary entropy and raw probabilities; FP32 only for small position chunks.

    MASK cannot be committed. Its probability is retained; probabilities are never
    renormalized after excluding MASK from the candidate ranking.
    """
    import torch
    result = []
    for start in range(0, len(positions), chunk_size):
        pp = positions[start:start+chunk_size]
        z = logits[pp].float()
        if not torch.isfinite(z).all():
            raise ValueError('Non-finite model logits')
        log_p = torch.log_softmax(z, dim=-1)
        probs = log_p.exp()
        entropy = -(probs * log_p).sum(dim=-1)
        raw_top = probs.argmax(dim=-1)
        mask_prob = probs[:, mask_id].clone()
        old_ids = [previous.get(p, {}).get('token_id', 0) for p in pp]
        old_probs = probs.gather(1, torch.tensor(old_ids, device=z.device)[:, None]).squeeze(1)
        probs[:, mask_id] = -1
        if max(2, top_k) >= probs.shape[-1]:
            raise ValueError('top-k must be smaller than vocabulary size')
        values, ids = probs.topk(max(2, top_k), dim=-1)
        ids, values = ids.cpu().tolist(), values.cpu().tolist()
        entropy, raw_top = entropy.cpu().tolist(), raw_top.cpu().tolist()
        mask_prob, old_probs = mask_prob.cpu().tolist(), old_probs.cpu().tolist()
        for j, p in enumerate(pp):
            old = previous.get(p)
            result.append({'position': p, 'token_id': ids[j][0], 'confidence': values[j][0],
                           'margin': values[j][0]-values[j][1], 'entropy': entropy[j],
                           'raw_top1_id': raw_top[j], 'mask_probability': mask_prob[j],
                           'topk_ids': ids[j][:top_k], 'topk_probs': values[j][:top_k],
                           'previous_confidence': old['confidence'] if old else None,
                           'delta_confidence': values[j][0]-old['confidence'] if old else None,
                           'previous_token_id': old['token_id'] if old else None,
                           'previous_token_probability_now': old_probs[j] if old else None,
                           'same_token_delta': old_probs[j]-old['confidence'] if old else None,
                           'prediction_changed': ids[j][0] != old['token_id'] if old else None})
    return result


def final_word_map(tokenizer, tokens):
    """Retrospective surface-word indices. Null where decoding prefixes are unstable."""
    import re
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    spans = [m.span() for m in re.finditer(r'\w+|[^\w\s]', text)]
    mapping, previous_end, valid = [], 0, True
    for i in range(len(tokens)):
        prefix = tokenizer.decode(tokens[:i+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not text.startswith(prefix) or len(prefix) < previous_end:
            valid = False
            break
        end = len(prefix)
        overlapping = [j for j, (a, b) in enumerate(spans) if a < end and b > previous_end]
        mapping.append(overlapping)
        previous_end = end
    return {'method': 'retrospective_unicode_words_and_punctuation', 'valid': valid,
            'token_word_indices': mapping if valid else None,
            'words': [text[a:b] for a, b in spans] if valid else None}


def collect_sample(model, tokenizer, config, sample, output):
    import torch
    # Preserve the dataset question verbatim; add no task or answer-format instructions.
    prompt_text = tokenizer.apply_chat_template([{'role': 'user', 'content': sample['question']}],
                                                tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if config['mask_id'] in prompt_ids:
        raise ValueError('Prompt contains MASK token')
    length, block = config['length'], config['block_length']
    context_limit = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'max_sequence_length', None)
    if context_limit and len(prompt_ids) + length > context_limit:
        raise ValueError(f'Prompt and response exceed context limit {context_limit}')
    x = torch.tensor([prompt_ids + [config['mask_id']] * length], device=model.device)
    pl = len(prompt_ids)
    specials = set(tokenizer.all_special_ids)
    stop_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    vocab = tokenizer.get_vocab()
    if '<|eot_id|>' in vocab:
        stop_ids.add(vocab['<|eot_id|>'])
    dictionary = {}
    previous, previous_metrics, last_commits = {}, None, []
    step, first_stop_step = 0, None
    start_time = time.perf_counter()
    partial = output / 'trace.jsonl.gz.partial'
    with gzip.open(partial, 'wt', encoding='utf-8', compresslevel=3) as handle, torch.inference_mode():
        emit(handle, {'type': 'header', 'schema_version': 1, 'config': config, 'sample': sample,
                      'prompt_protocol': PROMPT_PROTOCOL, 'user_message': sample['question'],
                      'prompt_text': prompt_text, 'prompt_ids': prompt_ids, 'stop_ids': sorted(stop_ids)})
        for block_start in range(0, length, block):
            block_end = min(length, block_start + block)
            local_step = 0
            while (x[0, pl+block_start:pl+block_end] == config['mask_id']).any().item():
                state = x[0, pl:].tolist()
                positions = [i for i, t in enumerate(state) if t == config['mask_id']]
                if x.is_cuda: torch.cuda.synchronize()
                t0 = time.perf_counter()
                response = model(x, use_cache=False)
                if x.is_cuda: torch.cuda.synchronize()
                forward_seconds = time.perf_counter() - t0
                rows = score_logits(response.logits[0, pl:], positions, previous, config['mask_id'], config['top_k'])
                del response
                filled = [i for i, t in enumerate(state) if t != config['mask_id']]
                observed_stop = min((i for i, t in enumerate(state) if t in stop_ids), default=None)
                for row in rows:
                    p, tid = row['position'], row['token_id']
                    for k in set(row['topk_ids'] + [row['raw_top1_id']]):
                        if str(k) not in dictionary:
                            dictionary[str(k)] = token_info(tokenizer, k, specials)
                    info = dictionary[str(tid)]
                    row.update({'eligible': block_start <= p < block_end,
                                'special': info['special'], 'category': info['category'],
                                'text': info['text'], **nearest(p, last_commits),
                                'nearest_filled_distance': min((abs(p-q) for q in filled), default=None),
                                'beyond_observed_stop': p > observed_stop if observed_stop is not None else None})
                quota = 1  # Window size is determined by offsets, not a quota.
                window_size = config.get('window_size', 3)
                chosen = select(rows, config['policy'], quota, config['commit_threshold'], window_size)
                committed = [r['position'] for r in chosen]
                selected = set(committed)
                for row in rows:
                    row['committed'] = row['position'] in selected
                metrics = step_metrics(rows, previous_metrics, last_commits, config['thresholds'])
                selection_metadata = {}
                if config['policy'] == 'top1_window':
                    anchor = committed[0]
                    selection_metadata = {'selection': {
                        'rule': ('top1_anchor_with_masked_offsets_minus1_plus1_plus2_v1' if window_size == 4
                                 else 'top1_anchor_with_masked_offsets_minus1_plus1_v2'),
                        'anchor_position': anchor,
                        'requested_offsets': [-1, 0, 1, 2] if window_size == 4 else [-1, 0, 1],
                        'actual_offsets': [p-anchor for p in committed],
                        'same_forward': True, 'neighbor_confidence_gate': None}}
                emit(handle, {'type': 'step', 'step': step, 'block_start': block_start, 'block_end': block_end,
                              'state_ids': state, 'previous_commits': last_commits,
                              'commit_positions': committed, 'commit_ids': [r['token_id'] for r in chosen],
                              'resolved_fraction': (length-len(rows))/length,
                              'forward_seconds': forward_seconds, 'metrics': metrics, 'positions': rows,
                              **selection_metadata})
                for row in chosen:
                    x[0, pl+row['position']] = row['token_id']
                if first_stop_step is None and any(r['token_id'] in stop_ids for r in chosen):
                    first_stop_step = step
                previous, previous_metrics, last_commits = {r['position']: r for r in rows}, metrics, committed
                step += 1
                local_step += 1
                if step % config['log_every'] == 0:
                    print(f"sample={sample['id']} step={step} filled={sum(t != config['mask_id'] for t in x[0, pl:].tolist())}/{length}", flush=True)
                if step > length:
                    raise RuntimeError('Decoder failed to make progress')
        tokens = x[0, pl:].tolist()
        end = min((i for i, t in enumerate(tokens) if t in stop_ids), default=length)
        answer = tokenizer.decode(tokens[:end], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predicted, method = numeric_answer(answer)
        gold, _ = numeric_answer(sample['answer'], reference=True)
        summary = {'type': 'result', 'sample_id': sample['id'], 'steps': step,
                   'question': sample['question'], 'reference': sample['answer'],
                   'final_ids': tokens, 'answer': answer, 'first_stop_position': end if end < length else None,
                   'first_stop_step': first_stop_step, 'answer_token_length': end,
                   'hit_length_limit': end == length, 'predicted_answer': predicted, 'gold_answer': gold,
                   'prompt_protocol': PROMPT_PROTOCOL, 'scoring_protocol': SCORING_PROTOCOL,
                   'answer_extraction': method, 'correct_numeric': predicted == gold and gold is not None,
                   # Keep legacy fields for old analysis consumers; strict is marker-only diagnostic.
                   'correct_strict': method == 'marked' and predicted == gold and gold is not None,
                   'correct_lenient': predicted == gold and gold is not None,
                   'elapsed_seconds': time.perf_counter()-start_time,
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated() if x.is_cuda else None, 'token_dictionary': dictionary,
                   'word_map': final_word_map(tokenizer, tokens)}
        emit(handle, summary)
    partial.replace(output / 'trace.jsonl.gz')
    dump(output / 'result.json', summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--revision', default=REVISION)
    p.add_argument('--dataset', default='openai/gsm8k')
    p.add_argument('--dataset-revision', default='main')
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--data-jsonl', type=Path, help='Offline rows containing question and answer (#### number)')
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--offset', type=int, default=0, help='Offset after deterministic dataset shuffle')
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--block-length', type=int, default=0, help='0 means full response')
    p.add_argument('--policy', choices=['top1', 'top1_window'], default='top1')
    p.add_argument('--window-size', type=int, choices=[3, 4], default=3,
                   help='Top1 window: 3 = offsets -1,0,+1; 4 = -1,0,+1,+2')
    p.set_defaults(commit_threshold=0.9)  # Pure measurement; neither policy gates on it.
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--mask-id', type=int, default=126336)
    p.add_argument('--log-every', type=int, default=32)
    return p


def main():
    args = parser().parse_args()
    config = vars(args).copy()
    out = config.pop('out')
    config['data_jsonl'] = str(args.data_jsonl.resolve()) if args.data_jsonl else None
    config['block_length'] = config['block_length'] or config['length']
    if min(args.samples, args.length, args.top_k, args.log_every) < 1 or args.offset < 0:
        raise ValueError('Counts must be positive; offset nonnegative')
    if config['block_length'] < 1 or args.length % config['block_length']:
        raise ValueError('block-length must divide length')
    if not all(0 <= x <= 1 for x in [args.commit_threshold] + args.thresholds):
        raise ValueError('Thresholds must lie in [0,1]')
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        execute(config, out)


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def execute(config, out, *, matched_samples=None, matched_dataset_info=None):
    import torch
    import numpy as np
    config = {**config, 'prompt_protocol': PROMPT_PROTOCOL, 'scoring_protocol': SCORING_PROTOCOL}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('source_sha256') != source_fingerprint():
            raise ValueError('Source changed since this run began; choose a fresh --out')
        if manifest['config'] != config:
            raise ValueError('Existing run has different arguments; choose a fresh --out')
        samples = [json.loads(s) for s in (out / 'samples.jsonl').read_text().splitlines()]
        dataset_info = manifest['dataset_info']
    else:
        if matched_samples is not None:
            samples = matched_samples
            dataset_info = matched_dataset_info
        elif config['data_jsonl']:
            source = Path(config['data_jsonl']).read_bytes()
            rows = [json.loads(s) for s in source.decode().splitlines() if s.strip()]
            dataset_info = {'sha256': hashlib.sha256(source).hexdigest(), 'source': config['data_jsonl']}
        else:
            from datasets import load_dataset
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info(config['dataset'], revision=config['dataset_revision']).sha
            ds = load_dataset(config['dataset'], 'main', split=config['split'], revision=revision)
            rows = list(ds)
            dataset_info = {'repository': config['dataset'], 'revision': revision, 'fingerprint': ds._fingerprint}
        if matched_samples is None:
            indices = list(range(len(rows)))
            random.Random(config['seed']).shuffle(indices)
            indices = indices[config['offset']:config['offset']+config['samples']]
            if len(indices) != config['samples']:
                raise ValueError('Requested sample range exceeds dataset')
            samples = [{'id': f'{i:05d}', 'dataset_index': i, 'question': rows[i]['question'], 'answer': rows[i]['answer']} for i in indices]
        if any(numeric_answer(s['answer'], reference=True)[0] is None for s in samples):
            raise ValueError('Dataset answers must contain a numeric #### reference')
        (out / 'samples.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False)+'\n' for s in samples))
        manifest = {'schema_version': 1, 'config': config, 'dataset_info': dataset_info,
                    'source_sha256': source_fingerprint(),
                    'created_unix': time.time(), 'python': platform.python_version(),
                    'git_commit': command('git', 'rev-parse', 'HEAD'),
                    'git_status': command('git', 'status', '--porcelain'),
                    'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    'nvidia_smi': command('nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv')}
        dump(manifest_path, manifest)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for s in samples:
        directory = out / 'samples' / s['id']
        if (directory / 'result.json').exists() and not (directory / 'trace.jsonl.gz').exists():
            raise ValueError(f'Completed result is missing its trace: {directory}')
    pending = [s for s in samples if not (out / 'samples' / s['id'] / 'result.json').exists()]
    if pending:
        model, tokenizer = load_model(config)
        for sample in pending:
            path = out / 'samples' / sample['id']
            path.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            result = collect_sample(model, tokenizer, config, sample, path)
            print(f"DONE sample={sample['id']} numeric_correct={result['correct_numeric']} extraction={result['answer_extraction']} steps={result['steps']} seconds={result['elapsed_seconds']:.1f}", flush=True)
    results = [json.loads((out / 'samples' / s['id'] / 'result.json').read_text()) for s in samples]
    dump(out / 'summary.json', {'sample_count': len(results),
                              'prompt_protocol': PROMPT_PROTOCOL, 'scoring_protocol': SCORING_PROTOCOL,
                              'numeric_accuracy': sum(r['correct_numeric'] for r in results)/len(results),
                              'primary_accuracy_metric': 'numeric_accuracy',
                              'strict_accuracy_note': 'Legacy marker-only diagnostic; not primary accuracy for question-only prompts',
                              'answer_extraction_counts': {method: sum(r['answer_extraction'] == method for r in results)
                                                           for method in sorted({r['answer_extraction'] for r in results})},
                              'strict_accuracy': sum(r['correct_strict'] for r in results)/len(results),
                              'lenient_accuracy': sum(r['correct_lenient'] for r in results)/len(results),
                              'length_limit_count': sum(r['hit_length_limit'] for r in results),
                              'total_steps': sum(r['steps'] for r in results)})


if __name__ == '__main__':
    main()
