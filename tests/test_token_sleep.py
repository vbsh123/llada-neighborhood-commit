"""CPU tensor fixtures only: no checkpoint, tokenizer, dataset, downloads or CUDA."""
from types import SimpleNamespace
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from experiments.token_sleep.simulation import FrozenStates, SleepSchedule, checked_blocks
from experiments.token_sleep.run import ProbeModel, evaluate
from confidence_geography.run import collect_sample


def row(position, entropy=3., confidence=.3):
    return {'position': position, 'entropy': entropy, 'confidence': confidence}


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.ffn = nn.Linear(8, 8)

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0., is_causal=False):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                             is_causal=is_causal)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = [z.reshape(b, n, 2, 4).transpose(1, 2) for z in self.qkv(x).chunk(3, -1)]
        # Position-dependent operation before interception verifies that the
        # cached tensors retain original full-sequence positions.
        pos = torch.arange(n, dtype=x.dtype).view(1, 1, n, 1)
        q, k = q+pos*.03, k+pos*.07
        att = self._scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, 8)
        hidden = x+att
        return hidden+torch.tanh(self.ffn(hidden)), None


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(), ToyBlock(), ToyBlock()])

    def forward(self, x):
        for block in self.blocks:
            x, _ = block(x)
        return x


class ScheduleTests(unittest.TestCase):
    def test_two_skips_then_wake_and_no_rescheduling_while_asleep(self):
        schedule = SleepSchedule(3, 'simulated_sleep')
        schedule.prepare(0, [9,9,9], 9)
        schedule.finish(0, [row(0), row(1), row(2)], [2])
        self.assertEqual(schedule.wake[:2], [3,3])
        self.assertEqual(schedule.prepare(1, [9,9,1], 9), [])  # all-asleep fallback
        # Keep one awake masked position to isolate timer behavior from fallback.
        schedule = SleepSchedule(3, 'simulated_sleep')
        schedule.prepare(0, [9,9,9], 9)
        schedule.finish(0, [row(0), row(1, confidence=.9), row(2)], [2])
        for step in [1,2]:
            self.assertEqual(schedule.prepare(step, [9,9,1], 9), [0])
            rr = [row(0, entropy=0., confidence=1.), row(1, confidence=.9)]
            schedule.finish(step, rr, [1])
            self.assertEqual(schedule.wake[0], 3)
            self.assertFalse(rr[0]['prediction_fresh'])
        self.assertEqual(schedule.prepare(3, [9,9,1], 9), [])

    def test_low_entropy_optional_middle_band_awake(self):
        schedule = SleepSchedule(4, 'delay_only', low_sleep=1)
        schedule.prepare(0, [9]*4, 9)
        schedule.finish(0, [row(0, .8), row(1, 1.5), row(2, 2.), row(3)], [3])
        self.assertEqual(schedule.wake, [2,1,3,1])

    def test_baseline_never_sleeps(self):
        schedule = SleepSchedule(2, 'baseline')
        schedule.prepare(0, [9,9], 9)
        self.assertEqual(schedule.finish(0, [row(0), row(1)], [0]), [])
        self.assertEqual(schedule.prepare(1, [1,9], 9), [])

    def test_fallback_wakes_everyone_and_committed_positions_cannot_sleep(self):
        schedule = SleepSchedule(2, 'simulated_sleep')
        schedule.wake = [10,10]
        self.assertEqual(schedule.prepare(2, [1,9], 9), [])
        self.assertTrue(schedule.fallback)
        self.assertEqual(schedule.wake[1], 2)


class FrozenStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = ToyModel().eval()
        self.x = torch.randn(1, 5, 8)

    def test_zero_sleep_bitwise_parity(self):
        with torch.inference_mode():
            raw = self.model(self.x)
            with FrozenStates(self.model.blocks) as frozen:
                frozen.begin([])
                actual = self.model(self.x)
                frozen.end()
            self.assertTrue(torch.equal(raw, actual))

    def test_stale_kv_consumed_outputs_frozen_bypass_and_wake(self):
        with torch.inference_mode(), FrozenStates(self.model.blocks) as frozen:
            frozen.begin([])
            initial = self.model(self.x)
            frozen.end()
            cache = [{k:v.clone() for k,v in c.items()} for c in frozen.cache]
            changed = self.x.clone()
            changed[:, 0] += 3.  # Revealed context; sleeper index 3 stays unchanged.
            with frozen.bypass():
                fresh = self.model(changed)
            for old, now in zip(cache, frozen.cache):
                for key in old:
                    self.assertTrue(torch.equal(old[key], now[key]))
            for _ in range(2):
                frozen.begin([3])
                asleep = self.model(changed)
                frozen.end()
                self.assertTrue(torch.equal(asleep[:,3], initial[:,3]))
                self.assertFalse(torch.equal(asleep[:,1], fresh[:,1]))
                for old, now in zip(cache, frozen.cache):
                    for key in ('k','v'):
                        self.assertTrue(torch.equal(old[key][:,:,3], now[key][:,:,3]))
                    self.assertTrue(torch.equal(old['output'][:,3], now['output'][:,3]))
                self.assertFalse(torch.equal(cache[0]['output'][:,0], frozen.cache[0]['output'][:,0]))
            frozen.begin([])
            awake = self.model(changed)
            frozen.end()
            self.assertTrue(torch.equal(awake, fresh))

    def test_kv_replacement_occurs_before_attention_consumption(self):
        block = self.model.blocks[0]
        original = block._scaled_dot_product_attention
        seen = []
        def record(q, k, v, **kwargs):
            seen.append((k.clone(), v.clone()))
            return original(q,k,v,**kwargs)
        block._scaled_dot_product_attention = record
        with torch.inference_mode(), FrozenStates(self.model.blocks) as frozen:
            frozen.begin([]); self.model(self.x); frozen.end()
            changed = self.x.clone(); changed[:,3] += 5.
            frozen.begin([3]); self.model(changed); frozen.end()
        self.assertTrue(torch.equal(seen[0][0][:,:,3], seen[1][0][:,:,3]))
        self.assertTrue(torch.equal(seen[0][1][:,:,3], seen[1][1][:,:,3]))

    def test_hooks_removed_on_exception(self):
        before = self.model.blocks[0]._scaled_dot_product_attention
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with FrozenStates(self.model.blocks):
                raise RuntimeError('fixture')
        self.assertEqual(before, self.model.blocks[0]._scaled_dot_product_attention)
        self.assertEqual(len(self.model.blocks[0]._forward_hooks), 0)

    def test_no_uninitialized_sleep(self):
        with torch.inference_mode(), FrozenStates(self.model.blocks) as frozen:
            frozen.begin([3])
            with self.assertRaisesRegex(ValueError, 'initialized cache'):
                self.model(self.x)

    def test_nonpinned_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            checked_blocks(self.model)


class FixtureTokenizer:
    all_special_ids = [0, 8, 9]
    eos_token_id = 8
    def apply_chat_template(self, messages, **kwargs): return messages[0]['content']
    def encode(self, text, **kwargs): return [9]
    def get_vocab(self): return {}
    def convert_ids_to_tokens(self, i): return str(i)
    def decode(self, ids, **kwargs): return ''.join(str(i) for i in ids if i not in self.all_special_ids)


class FixtureLM(ToyModel):
    device = 'cpu'
    config = SimpleNamespace(max_position_embeddings=64)
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 8)
        self.head = nn.Linear(8, 10)
        with torch.no_grad():
            self.head.weight.mul_(.01)
            self.head.bias.zero_()
    def forward(self, ids, use_cache=False):
        logits = self.head(super().forward(self.embedding(ids)))
        logits[:,1,1] += 8.
        logits[:,2,2] += 3.
        return SimpleNamespace(logits=logits)


class RunnerTests(unittest.TestCase):
    def test_shared_decoder_all_modes_and_counterfactual(self):
        self.exercise_decoder('top1')

    def test_shared_threshold_decoder_all_modes_and_counterfactual(self):
        self.exercise_decoder('threshold')

    def exercise_decoder(self, policy):
        config = dict(length=5, block_length=5, mask_id=0, top_k=3, policy=policy,
                      commit_threshold=.9, thresholds=[.9], log_every=100,
                      sleep_confidence=1., high_entropy=2., low_entropy=0.,
                      high_sleep=2, low_sleep=0, influence_every=1)
        # These cutoffs intentionally make many toy positions sleep. Normal
        # experimental cutoffs are tested separately in ScheduleTests.
        torch.manual_seed(321)
        model = FixtureLM().eval()
        tokenizer = FixtureTokenizer()
        sample = {'id': '00001', 'question': 'fixture', 'answer': '#### 1'}
        def cpu_forward(model, x): return model(x, use_cache=False), 0.
        with tempfile.TemporaryDirectory() as temp, patch('experiments.token_sleep.run.timed_forward', cpu_forward):
            out = Path(temp)
            for mode in ('baseline', 'delay_only', 'simulated_sleep'):
                directory = out/mode/'samples'/sample['id']; directory.mkdir(parents=True)
                context = FrozenStates(model.blocks) if mode == 'simulated_sleep' else nullcontext(None)
                with context as frozen:
                    probe = ProbeModel(model, config, mode, frozen)
                    self.assertIs(probe.config, model.config)
                    result = collect_sample(probe, tokenizer, config, sample, directory, select_output=probe.choose)
                    result['sleep_stats'] = dict(
                        sleep_position_forwards=probe.sleep_position_forwards,
                        masked_position_forwards=probe.masked_position_forwards,
                        all_asleep_full_wakes=probe.fallbacks,
                        treatment_forward_seconds=0., diagnostic_forward_seconds=0.,
                        diagnostic_forwards=probe.diagnostic_calls)
                    (directory/'result.json').write_text(json.dumps(result))
                self.assertEqual(result['steps'], 5)
                self.assertNotIn(0, result['final_ids'])
                with gzip.open(directory/'trace.jsonl.gz', 'rt') as handle:
                    steps = [r for r in map(json.loads, handle) if r['type'] == 'step']
                for step in steps:
                    sleepers = set(step['sleep']['sleep_positions'])
                    self.assertFalse(sleepers & set(step['commit_positions']))
                    if mode == 'simulated_sleep' and sleepers:
                        self.assertIsNotNone(step['sleep']['influence'])
            summary = evaluate(out, [sample])
            self.assertEqual(summary['modes']['baseline']['sleep_position_forwards'], 0)
            self.assertGreater(summary['modes']['simulated_sleep']['diagnostic_forwards'], 0)
            self.assertEqual(sum(summary['paired_vs_baseline']['simulated_sleep'].values()), 1)


if __name__ == '__main__':
    unittest.main()
