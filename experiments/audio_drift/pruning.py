"""One-shot, delayed prefix pruning; no future-answer information is used.

This is a quota intervention, not a calibrated low-attention safety rule.
Selection uses all heads and fixed query positions across the ANSWER CANVAS,
including masked positions. Selection/gather overhead is part of timed decoding.
"""
import math

import torch


def pruning_conditions(starts=(16, 32), fractions=(.25,), layer_start=24, queries=16, seed=1729):
    if (not starts or len(set(starts)) != len(starts) or any(s < 1 for s in starts)
            or not fractions or len(set(fractions)) != len(fractions)
            or any(not math.isfinite(f) or not 0 < f < 1 for f in fractions)
            or layer_start < 0 or queries < 1):
        raise ValueError('Invalid delayed pruning settings')
    specs = []
    for fraction in fractions:
        for start in starts:
            variants = [(scope, 'attention', 8) for scope in ('audio', 'prompt', 'prefix')]
            if start == max(starts):
                variants += [(scope, 'random', 8) for scope in ('audio', 'prompt', 'prefix')]
                variants += [(scope, 'attention', 1) for scope in ('audio', 'prompt', 'prefix')]
            for scope, selection, interval in variants:
                tag = format(fraction * 100, '.8g').replace('.', 'p')
                plan = dict(start=start, fraction=fraction, layer_start=layer_start,
                            queries=queries, seed=seed, scope=scope, selection=selection)
                specs.append(dict(condition=f'freeze_prefix_r{interval}_prune_{scope}_{selection}_f{tag}_start{start}_l{layer_start}',
                                  mode='freeze', schedule=[dict(group='prefix', refresh_every=interval)], pruning=plan))
    return specs


class LatePruning:
    def __init__(self, prepared, layers, specification):
        self.prepared = prepared
        self.spec = specification
        if specification is not None:
            p = specification
            pruning_conditions([p['start']], [p['fraction']], p['layer_start'], p['queries'], p['seed'])
            if p['layer_start'] >= layers or p['scope'] not in {'audio', 'prompt', 'prefix'} or p['selection'] not in {'attention', 'random'}:
                raise ValueError('Invalid pruning layer, scope or selection')
        self.layers = layers
        self.dropped, self.kept, self.records = {}, {}, []

    def select(self, layer, step, q, k, active):
        """Return retained absolute key positions; select once per eligible layer."""
        p = self.spec
        if p is None or layer < p['layer_start'] or step < p['start']:
            return None
        if layer not in self.dropped and layer > p['layer_start']:
            # A token removed at the pruning layer stays removed in every deeper
            # layer; never revive it using a frozen lower-layer representation.
            self.dropped[layer] = self.dropped[p['layer_start']]
            self.kept[layer] = self.kept[p['layer_start']]
        if layer not in self.dropped:
            audio = sorted(self.prepared.group_positions('both'))
            prompt = self.prepared.group_positions('prompt')
            pool = audio if p['scope'] == 'audio' else prompt if p['scope'] == 'prompt' else list(range(self.prepared.answer_start))
            # A common absolute token budget for all scopes, capped by the short
            # text group so audio/text/full-prefix comparisons are genuinely paired.
            count = max(0, min(len(audio) - 1, len(prompt) - 1, int(len(audio) * p['fraction'])))
            positions = torch.tensor(pool, dtype=torch.long, device=k.device)
            # Same fixed absolute answer-canvas query positions in every layer;
            # no baseline trajectory, EOS, or final answer span is consulted.
            offsets = torch.linspace(self.prepared.answer_start, k.shape[-2] - 1,
                                     min(p['queries'], k.shape[-2] - self.prepared.answer_start),
                                     device=q.device).long().unique()
            if active is None:
                active = torch.arange(k.shape[-2], device=q.device)
            local = torch.searchsorted(active, offsets)
            if not torch.equal(active.index_select(0, local), offsets):
                raise RuntimeError('Pruning must retain every answer query')
            probabilities = (q[:, :, local, :].float() @ k.float().transpose(-1, -2)
                             / math.sqrt(q.shape[-1])).softmax(-1)
            mean = probabilities.mean((0, 1, 2))
            if p['selection'] == 'attention':
                # Stable ties follow absolute sequence position.
                selected = torch.argsort(mean.index_select(0, positions), stable=True)[:count]
            else:
                generator = torch.Generator(device='cpu').manual_seed(p['seed'] + layer * 1009)
                selected = torch.randperm(len(pool), generator=generator)[:count].to(k.device)
            drop = positions.index_select(0, selected).sort().values
            removed = set(drop.cpu().tolist())
            self.dropped[layer] = tuple(sorted(removed))
            self.kept[layer] = torch.tensor([i for i in range(k.shape[-2]) if i not in removed],
                                            dtype=torch.long, device=k.device)
            self.records.append(dict(layer=layer, applies_to_layers=list(range(layer, self.layers)), step=step, dropped_positions=list(self.dropped[layer]),
                dropped_by_group={g: len(removed.intersection(self.prepared.group_positions(g)))
                                  for g in ('acoustic', 'semantic', 'prompt')},
                candidate_count=len(pool), requested_audio_fraction=p['fraction'], drop_budget=count,
                actual_audio_budget_fraction=count / max(1, len(audio)), actual_scope_fraction=count / max(1, len(pool)), probe_positions=offsets.cpu().tolist(),
                removed_attention_mass=float(mean.index_select(0, drop).sum())))
        return self.kept[layer]

    def accounting(self):
        return dict(specification=self.spec, activated_layers=len(self.dropped), selections=self.records,
                    note='One-shot quota selection; step is zero-based. Original RoPE positions and all answer rows retained. '
                         'Selected keys/values are omitted from attention immediately; their query/MLP rows are skipped on subsequent forwards. '
                         'Full-size backing banks, embedding/final norm/vocabulary allocations remain. '
                         'Attention probes span the full answer canvas, including masks, unlike retrospective diagnostic probes. '
                         'No inference about pruning safety follows from attention score alone.')
