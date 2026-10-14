"""Matched GSM8K baseline/delay/frozen-context experiment; CUDA only, no speed claim."""
import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

from confidence_geography.core import numeric_answer, select
from confidence_geography.run import (MODEL, REVISION, PROMPT_PROTOCOL, SCORING_PROTOCOL,
                                     collect_sample, dump, load_model, run_lock, command,
                                     score_logits)
from .simulation import FrozenStates, SleepSchedule, checked_blocks

HERE = Path(__file__).resolve().parent
MODES = ('baseline', 'delay_only', 'simulated_sleep')


def timed_forward(model, x):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = model(x, use_cache=False)
    torch.cuda.synchronize()
    return result, time.perf_counter()-start


class ProbeModel:
    """The shared decoder owns input/commit semantics; this wrapper owns sleep state."""

    def __init__(self, model, config, mode, frozen=None):
        self.model, self.settings, self.mode, self.frozen = model, config, mode, frozen
        self.schedule = SleepSchedule(config['length'], mode,
            confidence=config['sleep_confidence'], high_entropy=config['high_entropy'],
            low_entropy=config['low_entropy'], high_sleep=config['high_sleep'], low_sleep=config['low_sleep'])
        self.step = 0
        self.reference_logits = None
        self.actual_seconds = self.reference_seconds = 0.
        self.diagnostic_calls = 0
        self.sleep_position_forwards = self.masked_position_forwards = 0
        self.fallbacks = 0

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    def __call__(self, x, use_cache=False):
        if use_cache:
            raise ValueError('Autoregressive KV cache is disabled')
        pl = x.shape[1]-self.settings['length']
        state = x[0, pl:].tolist()
        sleepers = self.schedule.prepare(self.step, state, self.settings['mask_id'])
        self.sleep_position_forwards += len(sleepers)
        self.masked_position_forwards += sum(t == self.settings['mask_id'] for t in state)
        self.fallbacks += int(self.schedule.fallback)
        self.reference_logits = None
        self.this_reference_seconds = 0.
        interval = self.settings['influence_every']
        if self.mode == 'simulated_sleep' and sleepers and interval and self.step % interval == 0:
            # Crucial: the fresh control neither reads nor updates the sleep cache.
            with self.frozen.bypass():
                reference, seconds = timed_forward(self.model, x)
            self.reference_logits = reference.logits[0, pl:]
            self.reference_seconds += seconds
            self.this_reference_seconds = seconds
            self.diagnostic_calls += 1
        if self.frozen:
            self.frozen.begin([pl+p for p in sleepers])
        response, seconds = timed_forward(self.model, x)
        if self.frozen:
            self.frozen.end()
        self.this_actual_seconds = seconds
        self.actual_seconds += seconds
        return response

    def choose(self, rows, config, stop_ids):
        sleeping = set(self.schedule.sleepers)
        for row in rows:
            row['eligible'] = row['position'] not in sleeping
        chosen = select(rows, config['policy'], 1, config['commit_threshold'])
        if not chosen:
            raise RuntimeError('No fresh commitment after the all-asleep progress fallback')
        diagnostics = None
        if self.reference_logits is not None:
            positions = [r['position'] for r in rows]
            fresh = score_logits(self.reference_logits, positions, {}, config['mask_id'], config['top_k'])
            for row in fresh:
                row['eligible'] = True
            fresh_all = select(fresh, config['policy'], 1, config['commit_threshold'])
            for row in fresh:
                row['eligible'] = row['position'] not in sleeping
            fresh_awake = select(fresh, config['policy'], 1, config['commit_threshold'])
            fresh_map = {r['position']: r for r in fresh}
            awake = [r for r in rows if r['eligible']]
            actual_set = {(r['position'], r['token_id']) for r in chosen}
            fresh_set = {(r['position'], r['token_id']) for r in fresh_awake}
            diagnostics = {
                'scope': 'same_input_on_simulated_sleep_trajectory',
                'awake_count': len(awake),
                'awake_top1_token_changed_count': sum(r['token_id'] != fresh_map[r['position']]['token_id'] for r in awake),
                'awake_top1_confidence_absolute_change_sum': sum(abs(r['confidence']-fresh_map[r['position']]['confidence']) for r in awake),
                'awake_commit_set_changed': actual_set != fresh_set,
                'awake_commit_set_jaccard': len(actual_set & fresh_set)/len(actual_set | fresh_set),
                'fresh_unrestricted_commits': [[r['position'], r['token_id']] for r in fresh_all],
                'fresh_awake_commits': [[r['position'], r['token_id']] for r in fresh_awake],
                'fresh_unrestricted_sleeping_commit_count': sum(r['position'] in sleeping for r in fresh_all),
                'fresh_masked_predictions': [
                    {'position': r['position'], 'token_id': r['token_id'],
                     'confidence': r['confidence'], 'entropy': r['entropy']}
                    for r in fresh],
            }
            self.reference_logits = None
        assigned = self.schedule.finish(self.step, rows, [r['position'] for r in chosen])
        metadata = {'sleep': {
            'mode': self.mode, 'sleep_positions': sorted(sleeping),
            'assigned': assigned, 'all_asleep_full_wake': self.schedule.fallback,
            'treatment_forward_seconds': self.this_actual_seconds,
            'diagnostic_forward_seconds': self.this_reference_seconds,
            'influence': diagnostics}}
        self.step += 1
        return chosen, metadata


def fingerprint():
    base = HERE.parents[1]/'confidence_geography'
    paths = [*HERE.glob('*.py'), base/'run.py', base/'core.py']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(paths))).hexdigest()


def load_samples(args):
    if args.samples_from:
        source = args.samples_from.read_bytes()
        rows = [json.loads(s) for s in source.decode().splitlines() if s.strip()]
        samples = rows[args.offset:args.offset+args.samples]
        info = {'source': str(args.samples_from.resolve()), 'sha256': hashlib.sha256(source).hexdigest(),
                'selection': 'existing file order; no reshuffle'}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info('openai/gsm8k', revision='main').sha
        data = load_dataset('openai/gsm8k', 'main', split='test', revision=revision)
        indices = list(range(len(data)))
        random.Random(args.seed).shuffle(indices)
        samples = [{'id': f'{i:05d}', 'dataset_index': i, 'question': data[i]['question'],
                    'answer': data[i]['answer']} for i in indices[args.offset:args.offset+args.samples]]
        info = {'repository': 'openai/gsm8k', 'split': 'test', 'revision': revision,
                'selection': 'Python random.Random(seed) shuffle, then offset/count'}
    if len(samples) != args.samples or len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Wrong sample count or duplicate sample IDs')
    for sample in samples:
        if not str(sample['id']).isalnum() or not sample['question'].strip():
            raise ValueError('Invalid sample ID/question')
        if numeric_answer(sample['answer'], reference=True)[0] is None:
            raise ValueError('Reference must contain a numeric #### answer')
    return samples, info


def evaluate(out, samples):
    summary = {'samples': len(samples), 'modes': {}, 'paired_vs_baseline': {},
               'note': 'Simulation computes discarded work. Timings are NOT acceleration measurements. Full response including post-stop positions is filled.'}
    results = {}
    for mode in MODES:
        results[mode] = [json.loads((out/mode/'samples'/s['id']/'result.json').read_text()) for s in samples]
        rr = results[mode]
        asleep = sum(r['sleep_stats']['sleep_position_forwards'] for r in rr)
        masked = sum(r['sleep_stats']['masked_position_forwards'] for r in rr)
        influence = {'diagnosed_forwards': 0, 'awake_commit_set_changed_forwards': 0,
                     'awake_predictions': 0, 'awake_top1_token_changed_count': 0,
                     'awake_top1_confidence_absolute_change_sum': 0.,
                     'fresh_unrestricted_sleeping_commit_count': 0}
        if mode == 'simulated_sleep':
            for sample in samples:
                with gzip.open(out/mode/'samples'/sample['id']/'trace.jsonl.gz', 'rt') as handle:
                    for line in handle:
                        record = json.loads(line)
                        d = record.get('sleep', {}).get('influence')
                        if not d:
                            continue
                        influence['diagnosed_forwards'] += 1
                        influence['awake_commit_set_changed_forwards'] += int(d['awake_commit_set_changed'])
                        influence['awake_predictions'] += d['awake_count']
                        for key in ('awake_top1_token_changed_count', 'awake_top1_confidence_absolute_change_sum',
                                    'fresh_unrestricted_sleeping_commit_count'):
                            influence[key] += d[key]
            n = influence['awake_predictions']
            influence['awake_top1_token_changed_fraction'] = influence['awake_top1_token_changed_count']/n if n else None
            influence['mean_awake_top1_confidence_absolute_change'] = influence['awake_top1_confidence_absolute_change_sum']/n if n else None
            n = influence['diagnosed_forwards']
            influence['awake_commit_set_changed_fraction'] = influence['awake_commit_set_changed_forwards']/n if n else None
        summary['modes'][mode] = {
            'numeric_accuracy': sum(r['correct_numeric'] for r in rr)/len(rr),
            'generation_forwards': sum(r['steps'] for r in rr),
            'diagnostic_forwards': sum(r['sleep_stats']['diagnostic_forwards'] for r in rr),
            'sleep_position_forwards': asleep, 'masked_position_forwards': masked,
            'sleep_fraction_of_masked_position_forwards': asleep/masked if masked else 0.,
            'all_asleep_full_wakes': sum(r['sleep_stats']['all_asleep_full_wakes'] for r in rr),
            'treatment_forward_seconds': sum(r['sleep_stats']['treatment_forward_seconds'] for r in rr),
            'diagnostic_forward_seconds': sum(r['sleep_stats']['diagnostic_forward_seconds'] for r in rr),
            'influence': influence if mode == 'simulated_sleep' else None}
    for mode in MODES[1:]:
        counts = {'both_correct': 0, 'both_wrong': 0, 'baseline_only_correct': 0, 'treatment_only_correct': 0}
        for baseline, treatment in zip(results['baseline'], results[mode]):
            a, b = baseline['correct_numeric'], treatment['correct_numeric']
            key = 'both_correct' if a and b else 'both_wrong' if not a and not b else 'baseline_only_correct' if a else 'treatment_only_correct'
            counts[key] += 1
        summary['paired_vs_baseline'][mode] = counts
    dump(out/'comparison.json', summary)
    return summary


def run(args):
    import torch
    import numpy as np
    if min(args.samples, args.length, args.log_every) < 1 or args.offset < 0 or args.influence_every < 0:
        raise ValueError('Invalid counts')
    SleepSchedule(args.length, 'simulated_sleep', confidence=args.sleep_confidence,
                  high_entropy=args.high_entropy, low_entropy=args.low_entropy,
                  high_sleep=args.high_sleep, low_sleep=args.low_sleep)
    if not 0 <= args.commit_threshold <= 1:
        raise ValueError('Invalid commitment threshold')
    config = {k: v for k, v in vars(args).items() if k != 'out'}
    config['samples_from'] = str(args.samples_from.resolve()) if args.samples_from else None
    config.update(model=MODEL, revision=REVISION, mask_id=126336, block_length=args.length,
                  top_k=5, thresholds=[.5,.7,.8,.9,.95,.99],
                  prompt_protocol=PROMPT_PROTOCOL, scoring_protocol=SCORING_PROTOCOL,
                  experiment='simulated_token_sleep_v1')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        manifest_path = out/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['config'] != config or manifest['source_sha256'] != fingerprint():
                raise ValueError('Arguments/source changed; choose a new --out')
            source = (out/'samples.jsonl').read_bytes()
            if hashlib.sha256(source).hexdigest() != manifest['samples_sha256']:
                raise ValueError('Saved sample file changed')
            samples = [json.loads(s) for s in source.decode().splitlines()]
        else:
            if any(out.glob('*/samples/*/result.json')):
                raise ValueError('Existing results lack a manifest')
            print('Preparing matched GSM8K samples...', flush=True)
            samples, info = load_samples(args)
            source = ''.join(json.dumps(s, ensure_ascii=False)+'\n' for s in samples).encode()
            (out/'samples.jsonl').write_bytes(source)
            dump(manifest_path, {'config': config, 'dataset_info': info, 'source_sha256': fingerprint(),
                 'samples_sha256': hashlib.sha256(source).hexdigest(), 'git_commit': command('git', 'rev-parse', 'HEAD'),
                 'created_unix': time.time(), 'torch_version': torch.__version__,
                 'execution': 'Full computation with cache overwrites; no speed claim'})
        pending = []
        for sample in samples:
            for mode in MODES:
                directory = out/mode/'samples'/sample['id']
                if (directory/'result.json').exists():
                    if not (directory/'trace.jsonl.gz').exists():
                        raise ValueError('Completed result missing trace')
                    if 'sleep_stats' not in json.loads((directory/'result.json').read_text()):
                        # The shared collector writes result.json immediately
                        # before this runner adds its counters. Interrupted there:
                        # rerun this sample rather than accepting partial metadata.
                        pending.append((sample, mode, directory))
                else:
                    pending.append((sample, mode, directory))
        if pending:
            print(f'Loading pinned LLaDA; {len(samples)} questions × 3 modes; full window={args.length}. Simulation only.', flush=True)
            model, tokenizer = load_model(config)
            blocks = checked_blocks(model)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # Verify the untouched and instrumented zero-sleep paths on a real
            # input before any experimental output is accepted. Two extra calls.
            prompt_text = tokenizer.apply_chat_template([{'role': 'user', 'content': samples[0]['question']}],
                                                       tokenize=False, add_generation_prompt=True)
            prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
            ids = torch.tensor([prompt+[config['mask_id']]*args.length], device=model.device)
            with torch.inference_mode():
                ordinary, _ = timed_forward(model, ids)
                with FrozenStates(blocks) as frozen:
                    frozen.begin([])
                    instrumented, _ = timed_forward(model, ids)
                    frozen.end()
                difference = (ordinary.logits.float()-instrumented.logits.float()).abs().max().item()
                if difference != 0:
                    raise RuntimeError(f'Zero-sleep parity failed: max logit difference={difference}')
            dump(out/'startup_validation.json', {'zero_sleep_max_logit_difference': difference,
                 'extra_startup_forwards_this_process': 2, 'excluded_from_generation_counts': True})
            del ordinary, instrumented, ids
            for sample, mode, directory in pending:
                random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
                directory.mkdir(parents=True, exist_ok=True)
                torch.cuda.reset_peak_memory_stats()
                context = FrozenStates(blocks) if mode == 'simulated_sleep' else nullcontext(None)
                with context as frozen:
                    probe = ProbeModel(model, config, mode, frozen)
                    result = collect_sample(probe, tokenizer, config, sample, directory,
                                            select_output=probe.choose)
                    result['sleep_stats'] = {
                        'mode': mode, 'sleep_position_forwards': probe.sleep_position_forwards,
                        'masked_position_forwards': probe.masked_position_forwards,
                        'all_asleep_full_wakes': probe.fallbacks,
                        'treatment_forward_seconds': probe.actual_seconds,
                        'diagnostic_forward_seconds': probe.reference_seconds,
                        'diagnostic_forwards': probe.diagnostic_calls}
                    # result.json contains probe counters; trace result retains
                    # shared decoder fields. Per-step sleep diagnostics are in trace.
                    dump(directory/'result.json', result)
                print(f"DONE mode={mode} sample={sample['id']} correct={result['correct_numeric']} forwards={result['steps']} asleep_position_forwards={probe.sleep_position_forwards}", flush=True)
                del probe
            del model
        print(json.dumps(evaluate(out, samples), indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=10)
    p.add_argument('--samples-from', type=Path, help='Existing samples.jsonl in its saved order; no reshuffle')
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--policy', choices=['top1', 'threshold'], default='top1')
    p.add_argument('--commit-threshold', type=float, default=.9)
    p.add_argument('--sleep-confidence', type=float, default=.5)
    p.add_argument('--high-entropy', type=float, default=2.)
    p.add_argument('--low-entropy', type=float, default=1.)
    p.add_argument('--high-sleep', type=int, default=2)
    p.add_argument('--low-sleep', type=int, default=0, help='0 disables low-entropy sleep in the first experiment')
    p.add_argument('--influence-every', type=int, default=1, help='Fresh control every N steps with sleepers; 0 disables')
    p.add_argument('--log-every', type=int, default=32)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
