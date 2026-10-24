"""MARS-inspired temporal semantic chunks with first-forward global anchors.

Uses separate SDPA calls on gathered Q/K/V subsets, not a dense NxN mask.
This is a paper-based audio adaptation, not a reproduction of released MARS code.
No tokens are deleted. Acoustic summary queries and semantic anchor queries stay
global. Prompt/answer queries see all non-semantic keys plus semantic anchors.
Other semantic queries additionally see their own and neighboring time chunks.
"""
from contextlib import AbstractContextManager
import math

import torch


class AudioSparseAttention(AbstractContextManager):
    def __init__(self, llm, prepared, length, *, cache=None, chunk_size=32,
                 dense_layers=8, layer_group=8, anchor_fractions=(.5, .25, .125),
                 proxy_queries=128, neighbors=1):
        self.blocks = list(llm.model.transformer.blocks)
        if (llm.training or chunk_size < 1 or proxy_queries < 1 or layer_group < 1
                or not 0 <= dense_layers < len(self.blocks) or neighbors < 0
                or not anchor_fractions or any(not math.isfinite(f) or not 0 < f <= 1 for f in anchor_fractions)
                or len(anchor_fractions) != math.ceil((len(self.blocks)-dense_layers)/layer_group)):
            raise ValueError('Invalid audio sparse-attention settings')
        self.prepared, self.cache = prepared, cache
        self.length = prepared.answer_start + length
        self.chunk_size, self.neighbors = chunk_size, neighbors
        self.dense_layers, self.layer_group = dense_layers, layer_group
        self.fractions, self.proxy_queries = tuple(anchor_fractions), proxy_queries
        self.semantic = prepared.group_positions('semantic')
        if not self.semantic:
            raise ValueError('Sparse attention requires semantic audio tokens')
        self.chunks = [tuple(self.semantic[i:i+chunk_size]) for i in range(0, len(self.semantic), chunk_size)]
        self.anchors, self.patterns, self.originals = {}, {}, []
        self.step, self.pairs, self.dense_pairs, self.calls = -1, 0, 0, 0

    def search(self, q, k, group):
        # Only first forward, before any sparsity or cache reuse: all rows exist.
        if q.shape[-2] != self.length or k.shape[-2] != self.length:
            raise RuntimeError('Anchor search requires a full first-forward canvas')
        ids = torch.linspace(0, self.length-1, min(self.proxy_queries, self.length), device=q.device).long().unique()
        semantic = torch.tensor(self.semantic, device=q.device)
        score = k.new_zeros(len(semantic), dtype=torch.float32)
        keys = k.index_select(-2, semantic).float().transpose(-1, -2)
        for queries in ids.split(32):
            logits = q.index_select(-2, queries).float() @ keys / math.sqrt(q.shape[-1])
            logits.masked_fill_(queries[:, None] == semantic[None, :], -torch.inf)
            # One-semantic-token corner case: a self query has no allowed key.
            logits = torch.where(torch.isneginf(logits).all(-1, keepdim=True), torch.zeros_like(logits), logits)
            score += logits.softmax(-1).sum((0, 1, 2))
        anchors = []
        for start in range(0, len(self.semantic), self.chunk_size):
            stop = min(start+self.chunk_size, len(self.semantic))
            count = max(1, math.ceil((stop-start)*self.fractions[group]))
            picked = torch.argsort(score[start:stop], descending=True, stable=True)[:count] + start
            anchors.extend(semantic.index_select(0, picked).cpu().tolist())
        self.anchors[group] = tuple(sorted(anchors))

    def partitions(self, q_positions, group, device):
        # Query sets are disjoint and exhaust every active row. Key sets are
        # sorted, unique absolute positions, so cached RoPE keys remain valid.
        key = (group, tuple(q_positions), device)
        if key in self.patterns:
            return self.patterns[key]
        anchors = set(self.anchors[group])
        semantic = set(self.semantic)
        acoustic = set(self.prepared.group_positions('acoustic'))
        nonsemantic = set(range(self.length))-semantic
        lookup = {p: i for i, p in enumerate(q_positions)}
        groups = []

        def add(positions, keys):
            local = [lookup[p] for p in sorted(positions) if p in lookup]
            if local:
                groups.append((torch.tensor(local, device=device), torch.tensor(sorted(keys), device=device)))

        add(anchors | acoustic, set(range(self.length)))
        add(nonsemantic-acoustic, nonsemantic | anchors)
        for i, chunk in enumerate(self.chunks):
            neighborhood = set().union(*map(set, self.chunks[max(0,i-self.neighbors):i+self.neighbors+1]))
            add(set(chunk)-anchors, nonsemantic | anchors | neighborhood)
        if sum(len(rows) for rows, _ in groups) != len(q_positions):
            raise RuntimeError('Sparse query partition is incomplete or overlaps')
        self.patterns[key] = groups
        return groups

    def __enter__(self):
        try:
            for layer, block in enumerate(self.blocks):
                original = block._scaled_dot_product_attention
                self.originals.append((block, '_scaled_dot_product_attention' in block.__dict__,
                                       block.__dict__.get('_scaled_dot_product_attention')))

                def attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False,
                              _layer=layer, _original=original):
                    if attn_mask is not None or dropout_p or is_causal or q.shape[1] != k.shape[1]:
                        raise ValueError('Sparse adapter requires unmasked bidirectional equal-head inference')
                    if _layer == 0:
                        self.step += 1
                    self.dense_pairs += q.shape[-2] * k.shape[-2]
                    group = (_layer-self.dense_layers)//self.layer_group
                    if self.step == 0 and _layer >= self.dense_layers and (_layer-self.dense_layers)%self.layer_group == 0:
                        self.search(q, k, group)
                    if self.step == 0 or _layer < self.dense_layers:
                        self.pairs += q.shape[-2] * k.shape[-2]
                        self.calls += 1
                        return _original(q, k, v, attn_mask=None, dropout_p=0., is_causal=False)
                    if k.shape[-2] != self.length:
                        raise ValueError('Sparse attention cannot be combined with token deletion')
                    positions = (self.cache.active_ids_by_layer[_layer] if self.cache is not None
                                 else tuple(range(self.length)))
                    if len(positions) != q.shape[-2]:
                        raise RuntimeError('Sparse query coordinate mismatch')
                    parts = self.partitions(positions, group, q.device)
                    output = torch.empty_like(q)
                    for queries, keys in parts:
                        result = _original(q.index_select(-2, queries), k.index_select(-2, keys),
                                           v.index_select(-2, keys), attn_mask=None, dropout_p=0., is_causal=False)
                        output.index_copy_(-2, queries, result)
                        self.pairs += len(queries)*len(keys)
                        self.calls += 1
                    return output

                block._scaled_dot_product_attention = attention
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for block, existed, original in reversed(self.originals):
            if existed:
                block._scaled_dot_product_attention = original
            else:
                delattr(block, '_scaled_dot_product_attention')
        self.originals.clear()

    def accounting(self):
        return dict(forwards=self.step+1, attention_query_key_pairs=self.pairs,
                    dense_attention_pairs_at_same_active_queries=self.dense_pairs,
                    skipped_attention_pair_fraction=1-self.pairs/max(1,self.dense_pairs), sdpa_calls=self.calls,
                    anchors_by_group={str(g): list(v) for g, v in self.anchors.items()},
                    chunk_size=self.chunk_size, dense_layers=self.dense_layers, layer_group=self.layer_group,
                    anchor_fractions=self.fractions, proxy_queries=self.proxy_queries, neighbors=self.neighbors,
                    note='Pairs exclude anchor-search overhead; measured decode time includes it, gathers/scatters '
                         'and all small SDPA calls. Every hidden/QKV/MLP row remains unless the separate cache skips it. '
                         'Attention pair reduction is not a FLOP or speedup claim.')
