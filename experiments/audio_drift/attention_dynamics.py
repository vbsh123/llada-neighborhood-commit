"""Untimed baseline attention/representation diagnostics with bounded probe queries.

No attention outputs or decoder decisions are replaced. Probabilities are FP32
reconstructions for selected queries, not a claim of bitwise SDPA equivalence.
"""
from contextlib import AbstractContextManager
import json
import math

import numpy as np
import torch

from .hooks import blocks_of
from .metrics import drift


def correlations(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    def corr(a, b):
        if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    return {'count': len(x), 'pearson': corr(x, y),
            'spearman': corr(rankdata(x), rankdata(y))}


def js_distance(p, q):
    """Jensen-Shannon divergence, natural log, final axis; zero masses are safe."""
    m = (p + q) / 2
    tiny = torch.finfo(p.dtype).tiny
    def kl(a):
        return torch.where(a > 0, a * (a.clamp_min(tiny).log() - m.clamp_min(tiny).log()), 0).sum(-1)
    return (kl(p) + kl(q)) / 2


class AttentionDynamics(AbstractContextManager):
    def __init__(self, llm, prepared, layers, query_positions, work, save_tokens=False):
        self.blocks = blocks_of(llm)
        self.prepared, self.layers = prepared, set(layers)
        self.queries = list(query_positions)
        if not self.layers or not self.layers <= set(range(len(self.blocks))):
            raise ValueError('Invalid diagnostic layers')
        if not self.queries or len(set(self.queries)) != len(self.queries):
            raise ValueError('Unique nonempty probe positions required')
        self.work, self.save_tokens = work, save_tokens
        self.stream = (work / 'attention_dynamics.jsonl').open('w')
        self.previous, self.pending, self.originals, self.handles = {}, {}, [], []
        self.positions = {g: prepared.group_positions(g) for g in ('acoustic', 'prompt', 'semantic')}
        self.labels = {'acoustic': 'audio_acoustic', 'prompt': 'prompt', 'semantic': 'audio_semantic'}

    def begin(self, meta, x):
        self.meta = dict(meta)
        self.ids = x[0, [p - self.prepared.audio_count for p in self.queries]].detach().cpu()
        self.pending.clear()
        self.arrays = {}

    def __enter__(self):
        try:
            for layer in sorted(self.layers):
                block = self.blocks[layer]
                name = '_scaled_dot_product_attention'
                existed = name in block.__dict__
                self.originals.append((block, name, existed, block.__dict__.get(name)))
                original = getattr(block, name)
                def attention(q, k, v, _layer=layer, _original=original, **kwargs):
                    if (q.shape[0] != 1 or q.shape[-2] != k.shape[-2] or
                            q.shape[1] != k.shape[1] or kwargs.get('attn_mask') is not None or
                            kwargs.get('is_causal', False) or kwargs.get('dropout_p', 0)):
                        raise ValueError('Diagnostics require pinned batch-one, full bidirectional attention')
                    output = _original(q, k, v, **kwargs)
                    selected = q[0, :, self.queries, :].float()
                    probabilities = (selected @ k[0].float().transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
                    self.pending[_layer] = {
                        'probabilities': probabilities.detach(), 'query': selected.detach(),
                        'key': k[0, :, :self.prepared.answer_start].transpose(0, 1).flatten(1).detach().clone(),
                        'value': v[0, :, :self.prepared.answer_start].transpose(0, 1).flatten(1).detach().clone(),
                        'contributions': {g: (probabilities[..., pos] @ v[0, :, pos, :].float()).detach()
                                          for g, pos in self.positions.items()}}
                    return output
                setattr(block, name, attention)
                def hidden(module, inputs, output, _layer=layer):
                    self.observe(_layer, output[0][0, :self.prepared.answer_start])
                self.handles.append(block.register_forward_hook(hidden))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def observe(self, layer, hidden):
        packet = self.pending.pop(layer)
        packet['hidden'] = hidden.detach().clone()
        packet['ids'] = self.ids.clone()
        old = self.previous.get(layer)
        if old is not None:
            common = self.ids == old['ids']
            common_device = common.to(hidden.device)
            current_p = packet['probabilities'][:, common_device]
            previous_p = old['probabilities'][:, common_device]
            arrays = {kind: drift(packet[kind], old[kind]) for kind in ('hidden', 'key', 'value')}
            for group, pos in self.positions.items():
                h = arrays['hidden']['cosine_distance'][pos]
                row = {**self.meta, 'layer': layer, 'group': self.labels[group], 'count': len(pos),
                       'query_count': len(self.queries), 'unchanged_query_count': int(common.sum()),
                       'unchanged_masked_query_count': int((common & (self.ids == self.prepared.mask_id)).sum()),
                       'hidden_cosine_distance': float(h.mean()),
                       'hidden_relative_l2': float(arrays['hidden']['relative_l2'][pos].mean()),
                       'key_cosine_distance': float(arrays['key']['cosine_distance'][pos].mean()),
                       'value_cosine_distance': float(arrays['value']['cosine_distance'][pos].mean())}
                if common.any():
                    p, q = current_p[..., pos], previous_p[..., pos]
                    mass, old_mass = p.sum(-1), q.sum(-1)
                    conditional_p = p / mass.unsqueeze(-1).clamp_min(1e-30)
                    conditional_q = q / old_mass.unsqueeze(-1).clamp_min(1e-30)
                    valid = (mass > 1e-20) & (old_mass > 1e-20)
                    change = (p - q).abs().mean((0, 1))
                    contribution = packet['contributions'][group][:, common_device].flatten(0, 1)
                    old_contribution = old['contributions'][group][:, common_device].flatten(0, 1)
                    change_out = drift(contribution, old_contribution)
                    query_change = drift(packet['query'][:, common_device].flatten(0, 1),
                                         old['query'][:, common_device].flatten(0, 1))
                    row.update(attention_mass=float(mass.mean()),
                               attention_mass_abs_change=float((mass - old_mass).abs().mean()),
                               attention_l1_change=float((p - q).abs().sum(-1).mean()),
                               conditional_attention_js=float(js_distance(conditional_p, conditional_q)[valid].mean()) if valid.any() else None,
                               contribution_relative_l2=float(change_out['relative_l2'].mean()),
                               contribution_cosine_distance=float(change_out['cosine_distance'].mean()),
                               query_cosine_distance=float(query_change['cosine_distance'].mean()),
                               token_drift_vs_attention_change=correlations(h.cpu().numpy(), change.cpu().numpy()))
                    if self.save_tokens:
                        prefix = f'l{layer:02d}_{group}'
                        self.arrays[prefix + '_positions'] = np.asarray(pos, dtype=np.int32)
                        self.arrays[prefix + '_hidden_cosine_distance'] = h.cpu().numpy()
                        self.arrays[prefix + '_attention_abs_change'] = change.cpu().numpy()
                        self.arrays[prefix + '_attention_mean'] = p.mean((0, 1)).cpu().numpy()
                self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.previous[layer] = packet

    def end(self):
        if self.pending:
            raise RuntimeError('Attention diagnostics missing block output hooks')
        self.stream.flush()
        if self.arrays:
            path = self.work / 'attention_token_metrics'
            path.mkdir(exist_ok=True)
            np.savez_compressed(path / f"{self.meta['step']:04d}.npz", **self.arrays)

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        for block, name, existed, value in reversed(self.originals):
            if existed:
                setattr(block, name, value)
            else:
                delattr(block, name)
        self.handles.clear(); self.originals.clear(); self.previous.clear(); self.pending.clear()
        self.stream.close()


@torch.no_grad()
def collect_dynamics(llm, prepared, states, events, baseline_ids, eos_id, args, work):
    from .decode import forward_logits
    from .run import write_json
    stop = eos_forward_count(events, baseline_ids, eos_id)
    content = baseline_ids.index(eos_id) if eos_id in baseline_ids else len(baseline_ids)
    # Fixed query positions chosen retrospectively within the visible answer span.
    # Selection is diagnostic only; no future information enters model inputs.
    query_offsets = sorted(set(np.linspace(0, max(0, content - 1), args.diagnostic_queries, dtype=int).tolist()))
    queries = [prepared.answer_start + p for p in query_offsets]
    layers = list(range(llm.config.n_layers)) if args.diagnostic_layers == 'all' else [int(v) for v in args.diagnostic_layers.split(',')]
    write_json(work / 'attention_dynamics_config.json', {
        'layers': layers, 'query_offsets': query_offsets, 'forwards': stop,
        'note': 'Separate untimed replay of uncached baseline states. All heads, fixed sampled answer queries. '
                'Only queries whose input identity is unchanged are compared. FP32 attention reconstruction; '
                'group contribution is before output projection. Not a causal importance measurement. '
                'Masked future answer slots are not included in the text/prompt group.'})
    print(f'  attention dynamics: {stop} baseline forwards; {len(layers)} layers; {len(queries)} fixed queries', flush=True)
    with AttentionDynamics(llm, prepared, layers, queries, work, args.diagnostic_tokens) as observer:
        revealed = set()
        content_positions = set(range(content))
        for step in range(stop):
            x = states[step].to(prepared.prompt.device)
            meta = {**events[step], 'retrospective_content_revealed_fraction':
                    len(revealed & content_positions) / len(content_positions) if content_positions else None}
            observer.begin(meta, x)
            output = forward_logits(llm, prepared, x)
            del output
            observer.end()
            revealed.update(events[step]['committed_positions'])
            if (step + 1) % 32 == 0:
                print(f'    attention diagnostics {step + 1}/{stop}', flush=True)


def eos_forward_count(events, answer_ids, eos_id):
    if eos_id not in answer_ids:
        return len(events)
    required = set(range(answer_ids.index(eos_id) + 1))
    visible = set()
    for count, event in enumerate(events, 1):
        visible.update(event['committed_positions'])
        if required <= visible:
            return count
    raise ValueError('Final EOS prefix was never committed')
