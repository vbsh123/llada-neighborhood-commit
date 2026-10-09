"""Independent GSM8K selection and question-only prompts for the SOAR probe."""
import hashlib
import json
from pathlib import Path
import random


def select_samples(rows, count, offset, seed):
    if count < 1 or offset < 0:
        raise ValueError('Require positive samples and nonnegative offset')
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    indices = indices[offset:offset+count]
    if len(indices) != count:
        raise ValueError('Requested sample range exceeds dataset')
    return [{'id': f'{i:05d}', 'dataset_index': i,
             'question': rows[i]['question'], 'answer': rows[i]['answer']} for i in indices]


def load_inputs(config, saved=None):
    """Resume uses our saved samples; no earlier experiment or dataset download."""
    if saved is not None:
        return [json.loads(line) for line in saved.read_text().splitlines() if line.strip()], None
    if config['data_jsonl']:
        source = Path(config['data_jsonl']).read_bytes()
        rows = [json.loads(line) for line in source.decode().splitlines() if line.strip()]
        info = {'source': config['data_jsonl'], 'sha256': hashlib.sha256(source).hexdigest()}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(config['dataset'], revision=config['dataset_revision']).sha
        dataset = load_dataset(config['dataset'], 'main', split=config['split'], revision=revision)
        rows = list(dataset)
        info = {'repository': config['dataset'], 'revision': revision,
                'split': config['split'], 'fingerprint': dataset._fingerprint}
    return select_samples(rows, config['samples'], config['offset'], config['seed']), info


def question_prompt(tokenizer, question, mask_id):
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                         tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if mask_id in ids:
        raise ValueError('Prompt contains MASK token')
    stops = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    vocabulary = tokenizer.get_vocab()
    if '<|eot_id|>' in vocabulary:
        stops.add(vocabulary['<|eot_id|>'])
    return {'prompt_text': text, 'prompt_ids': ids, 'stop_ids': sorted(stops)}
