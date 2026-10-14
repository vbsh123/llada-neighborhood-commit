"""Timers and temporary LLaDA hooks. Never delete positions or alter attention masks."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import types


LLADA_SOURCE_SHA256 = '98bac7e53fef0bb7ca01e3716c11a7f710d183e10dbb9783b88db9dbba2e3766'


def checked_blocks(model):
    """Fail closed if HF remote code differs from the source inspected for this probe."""
    source = Path(inspect.getfile(type(model)))
    if hashlib.sha256(source.read_bytes()).hexdigest() != LLADA_SOURCE_SHA256:
        raise ValueError('Unsupported LLaDA implementation: pinned modeling_llada.py hash differs')
    transformer = model.model.transformer
    if 'blocks' not in transformer:
        raise ValueError('This probe supports ungrouped LLaDA blocks only')
    blocks = list(transformer.blocks)
    if not blocks or any(type(b).__name__ != 'LLaDALlamaBlock' for b in blocks):
        raise ValueError('Expected the pinned LLaDALlamaBlock architecture')
    return blocks


class SleepSchedule:
    """Deadlines use response indices. A skip of 2 at t wakes at t+3."""

    def __init__(self, length, mode, *, confidence=.5, high_entropy=2.,
                 low_entropy=1., high_sleep=2, low_sleep=0):
        if mode not in ('baseline', 'delay_only', 'simulated_sleep'):
            raise ValueError('Unknown mode')
        if length < 1 or min(high_sleep, low_sleep) < 0:
            raise ValueError('Invalid counts')
        if not 0 <= confidence <= 1 or not 0 <= low_entropy <= high_entropy:
            raise ValueError('Invalid uncertainty cutoffs')
        self.mode, self.confidence = mode, confidence
        self.high_entropy, self.low_entropy = high_entropy, low_entropy
        self.high_sleep, self.low_sleep = high_sleep, low_sleep
        self.wake = [0] * length
        self.observed = [None] * length
        self.sleepers = []
        self.fallback = False

    def prepare(self, step, state, mask_id):
        masked = [p for p, token in enumerate(state) if token == mask_id]
        # Committed tokens never remain sleepers, regardless of their old timer.
        self.sleepers = [p for p in masked if self.wake[p] > step]
        self.fallback = bool(masked) and len(masked) == len(self.sleepers)
        if self.fallback:
            # Explicit full wake, not a stale-prediction commitment.
            for p in self.sleepers:
                self.wake[p] = step
            self.sleepers = []
        return self.sleepers

    def finish(self, step, rows, committed):
        sleeping = set(self.sleepers)
        committed = set(committed)
        assigned = []
        for row in rows:
            p = row['position']
            row['sleeping'] = p in sleeping
            # delay_only computes fresh counterfactual predictions for sleepers,
            # but does not use them to reschedule or commit.
            row['prediction_fresh'] = self.mode != 'simulated_sleep' or p not in sleeping
            row['observation_age_for_schedule'] = (step-self.observed[p]
                                                   if self.observed[p] is not None else None)
            if p not in sleeping:
                self.observed[p] = step
                skip = 0
                if self.mode != 'baseline' and p not in committed and row['confidence'] <= self.confidence:
                    if row['entropy'] >= self.high_entropy:
                        skip = self.high_sleep
                    elif row['entropy'] < self.low_entropy:
                        skip = self.low_sleep
                self.wake[p] = step + skip + 1
                if skip:
                    assigned.append({'position': p, 'skip_forwards': skip, 'wake_step': self.wake[p],
                                     'entropy': row['entropy'], 'confidence': row['confidence']})
            row['wake_step'] = self.wake[p]
        return assigned


class FrozenStates:
    """Compute normally, overwrite sleepers before consumption, retain all sequence rows.

    K/V interception is inside attention, AFTER head shaping and RoPE but BEFORE
    SDPA. Block hooks restore final residual/FFN outputs before the next block.
    This simulates selective execution; it performs and discards sleeping work.
    """

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.cache = [{} for _ in self.blocks]
        self.sleepers = []
        self.paused = False
        self.calls = None
        self.originals, self.handles = [], []

    def __enter__(self):
        if any(b.training for b in self.blocks):
            raise ValueError('Simulation requires eval mode')
        try:
            for index, block in enumerate(self.blocks):
                original = block._scaled_dot_product_attention
                self.originals.append((block, original))

                def attention(_block, q, k, v, attn_mask=None, dropout_p=0., is_causal=False,
                              _index=index, _original=original):
                    if self.paused:
                        return _original(q, k, v, attn_mask=attn_mask,
                                         dropout_p=dropout_p, is_causal=is_causal)
                    if self.calls is None or dropout_p or is_causal:
                        raise ValueError('Unexpected attention invocation')
                    if q.shape[0] != 1 or q.shape[-2] != k.shape[-2]:
                        raise ValueError('Only batch-one full-sequence attention is supported')
                    self.calls[_index][0] += 1
                    if self.sleepers:
                        cache = self.cache[_index]
                        if 'k' not in cache or cache['k'].shape != k.shape:
                            raise ValueError('Sleep attempted without a matching initialized cache')
                        k = k.clone()
                        v = v.clone()
                        k[:, :, self.sleepers, :] = cache['k'][:, :, self.sleepers, :]
                        v[:, :, self.sleepers, :] = cache['v'][:, :, self.sleepers, :]
                    self.cache[_index]['k'] = k.detach().clone()
                    self.cache[_index]['v'] = v.detach().clone()
                    return _original(q, k, v, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal)

                def output_hook(_block, inputs, output, _index=index):
                    if self.paused:
                        return output
                    if self.calls is None or not isinstance(output, tuple) or len(output) != 2:
                        raise ValueError('Unexpected block output')
                    hidden, present = output
                    if present is not None:
                        raise ValueError('Autoregressive use_cache/layer_past is unsupported')
                    self.calls[_index][1] += 1
                    if self.sleepers:
                        cached = self.cache[_index].get('output')
                        if cached is None or cached.shape != hidden.shape:
                            raise ValueError('Missing cached layer output')
                        hidden = hidden.clone()
                        hidden[:, self.sleepers, :] = cached[:, self.sleepers, :]
                    self.cache[_index]['output'] = hidden.detach().clone()
                    return hidden, present

                block._scaled_dot_product_attention = types.MethodType(attention, block)
                self.handles.append(block.register_forward_hook(output_hook))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def begin(self, absolute_sleepers):
        if self.calls is not None:
            raise RuntimeError('Previous forward was not finished')
        self.sleepers = list(absolute_sleepers)
        self.calls = [[0, 0] for _ in self.blocks]

    def end(self):
        calls, self.calls = self.calls, None
        if calls != [[1, 1] for _ in self.blocks]:
            raise RuntimeError(f'Expected exactly one attention/output call per block: {calls}')

    @contextmanager
    def bypass(self):
        old = self.paused
        self.paused = True
        try:
            yield
        finally:
            self.paused = old

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for block, original in self.originals:
            block._scaled_dot_product_attention = original
        self.handles.clear()
        self.originals.clear()
        self.cache.clear()
