"""Skip cached fixed-prefix rows through QKV, attention queries and feed-forward.

Inference-only adapter for the pinned, ungrouped LLaDALlamaBlock. All answer
rows remain active. Absolute RoPE positions are preserved. Cache-only policies
retain every KV column; optional delayed pruning removes selected prefix columns
and their subsequent transformer rows.
No upstream source changes.
"""
from contextlib import AbstractContextManager

import torch

from .hooks import blocks_of
from .protocol import layer_intervals,refresh_phases,refresh_due
from .pruning import LatePruning


class PrefixCache(AbstractContextManager):
    def __init__(self, llm, prepared, schedules, pruning=None):
        if llm.training:
            raise ValueError("PrefixCache requires eval mode")
        self.blocks = blocks_of(llm)
        self.prepared = prepared
        self.pruning = LatePruning(prepared, len(self.blocks), pruning)
        self.schedules = []
        self.phases = []
        claimed = set()
        for spec in schedules:
            positions = prepared.group_positions(spec["group"])
            if not positions or claimed.intersection(positions) or spec["refresh_every"] < 0:
                raise ValueError("Cache schedules must select nonempty disjoint fixed-prefix groups")
            claimed.update(positions)
            self.schedules.append((positions, layer_intervals(spec, len(self.blocks))))
            self.phases.append(refresh_phases(spec))
        self.step = -1
        self.patterns = {}
        self.pattern_ids = {}
        self.banks = {}
        self.events = []
        self.originals = []
        # Absolute query coordinates for optional structured-attention adapters.
        self.active_by_layer = {}
        self.active_ids_by_layer = {}

    def __enter__(self):
        try:
            for layer, block in enumerate(self.blocks):
                if type(block).__name__ != "LLaDALlamaBlock" or block._activation_checkpoint_fn is not None:
                    raise ValueError("Compact path supports pinned LLaDALlamaBlock without checkpointing only")
                if not block.config.rope:
                    raise ValueError("Absolute RoPE required")
                existed = "forward" in block.__dict__
                self.originals.append((block, existed, block.__dict__.get("forward")))
                original = block.forward

                def forward(x, attention_bias=None, layer_past=None, use_cache=False,
                            replace_position=None, _layer=layer, _block=block, _original=original):
                    if (x.shape[0] != 1 or attention_bias is not None or layer_past is not None
                            or use_cache or replace_position is not None):
                        raise ValueError("Compact path requires batch one, bidirectional no-past attention")
                    if _layer == 0:
                        self.step += 1
                        self.events.append({"step": self.step, "total_rows": x.shape[1],
                                            "layer_computed_rows": [], "layer_refresh": [], "layer_kv_columns": []})
                    # Each layer has its own clock and banks. Flat schedules
                    # expand to identical intervals at every layer.
                    due = tuple(refresh_due(self.step,intervals[_layer],phases)
                                for (_,intervals),phases in zip(self.schedules,self.phases))
                    removed = self.pruning.dropped.get(_layer, ())
                    key = (x.shape[1], x.device, due, removed)
                    if key not in self.patterns:
                        inactive = {p for (positions, _), refresh in zip(self.schedules, due)
                                    if not refresh for p in positions} | set(removed)
                        selected = [p for p in range(x.shape[1]) if p not in inactive]
                        self.patterns[key] = torch.tensor(selected, dtype=torch.long, device=x.device)
                        self.pattern_ids[key] = tuple(selected)
                    active = self.patterns[key]
                    self.active_by_layer[_layer] = active
                    self.active_ids_by_layer[_layer] = self.pattern_ids[key]
                    event = self.events[-1]
                    if _layer == 0:
                        # Historical fields describe layer zero; aggregate
                        # accounting below uses the full per-layer counts.
                        event.update(computed_rows=len(active), refresh=list(due))
                    event['layer_computed_rows'].append(len(active))
                    event['layer_refresh'].append(list(due))
                    if len(active) == x.shape[1]:
                        # Use upstream verbatim for full refresh and identity controls.
                        attention = _block._scaled_dot_product_attention

                        def capture(q, k, v, **kwargs):
                            self.banks[_layer] = [k.detach().clone(), v.detach().clone(), None]
                            keep = self.pruning.select(_layer, self.step, q, k, None)
                            event['layer_kv_columns'].append(k.shape[-2] if keep is None else len(keep))
                            if keep is not None:
                                k, v = k.index_select(-2, keep), v.index_select(-2, keep)
                            return attention(q, k, v, **kwargs)
                        _block._scaled_dot_product_attention = capture
                        try:
                            output = _original(x)
                        finally:
                            _block._scaled_dot_product_attention = attention
                        self.banks[_layer][2] = output[0].detach().clone()
                        return output
                    if _layer not in self.banks:
                        raise RuntimeError("Cache must be initialized by a full first forward")
                    y = x.index_select(1, active)
                    normalized = _block.attn_norm(y)
                    q, k, v = _block.q_proj(normalized), _block.k_proj(normalized), _block.v_proj(normalized)
                    dtype = k.dtype
                    if _block.q_norm is not None and _block.k_norm is not None:
                        q, k = _block.q_norm(q).to(dtype), _block.k_norm(k).to(dtype)
                    B, T, C = q.shape
                    q = q.view(B, T, _block.config.n_heads, C // _block.config.n_heads).transpose(1, 2)
                    k = k.view(B, T, _block.config.effective_n_kv_heads, C // _block.config.n_heads).transpose(1, 2)
                    v = v.view(B, T, _block.config.effective_n_kv_heads, C // _block.config.n_heads).transpose(1, 2)
                    rotary = _block.rotary_emb
                    q_rot, k_rot = (q.float(), k.float()) if _block.config.rope_full_precision else (q, k)
                    with torch.autocast(q.device.type, enabled=False):
                        sin, cos = rotary.get_rotary_embedding(x.shape[1], q.device)
                        sin, cos = sin.type_as(q_rot).index_select(-2, active), cos.type_as(q_rot).index_select(-2, active)
                        q_rot = rotary.apply_rotary_pos_emb(sin, cos, q_rot)
                        k_rot = rotary.apply_rotary_pos_emb(sin, cos, k_rot)
                    full_k, full_v, hidden = self.banks[_layer]
                    full_k.index_copy_(-2, active, k_rot.type_as(k))
                    full_v.index_copy_(-2, active, v)
                    keep = self.pruning.select(_layer, self.step, q_rot.type_as(q), full_k, active)
                    event['layer_kv_columns'].append(full_k.shape[-2] if keep is None else len(keep))
                    keys, values = ((full_k, full_v) if keep is None else
                                    (full_k.index_select(-2, keep), full_v.index_select(-2, keep)))
                    attended = _block._scaled_dot_product_attention(q_rot.type_as(q), keys, values,
                                      attn_mask=None, dropout_p=0., is_causal=False)
                    attended = attended.transpose(1, 2).contiguous().view(B, T, C)
                    y = y + _block.dropout(_block.attn_out(attended))
                    normalized = _block.ff_norm(y)
                    feed = _block.act(_block.ff_proj(normalized)) * _block.up_proj(normalized)
                    y = y + _block.dropout(_block.ff_out(feed))
                    hidden.index_copy_(1, active, y)
                    return hidden, None

                block.forward = forward
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for block, existed, original in reversed(self.originals):
            if existed:
                block.forward = original
            else:
                delattr(block, "forward")
        self.originals.clear()

    def accounting(self):
        if any(len(e['layer_computed_rows']) != len(self.blocks) or
               len(e['layer_kv_columns']) != len(self.blocks) for e in self.events):
            raise RuntimeError("Incomplete transformer forwards in cache accounting")
        full = sum(e["total_rows"] * len(self.blocks) for e in self.events)
        used = sum(sum(e['layer_computed_rows']) for e in self.events)
        result = {"forwards": len(self.events), "computed_token_layers": used,
                "dense_token_layers_at_same_forward_count": full,
                "pruning": self.pruning.accounting(),
                "kv_columns_by_layer": [sum(e['layer_kv_columns'][i] for e in self.events)
                                        for i in range(len(self.blocks))],
                "qk_query_key_pairs_by_layer": [sum(e['layer_computed_rows'][i] * e['layer_kv_columns'][i]
                                                     for e in self.events) for i in range(len(self.blocks))],
                "computed_rows_by_layer": [sum(e['layer_computed_rows'][i] for e in self.events)
                                           for i in range(len(self.blocks))],
                "effective_schedules": [{"positions": len(positions), "refresh_by_layer": list(intervals)}
                                        for positions, intervals in self.schedules],
                "skipped_token_update_fraction": 1 - used / max(1, full),
                "note": "Counts skipped QKV, attention query/output and MLP rows; not total FLOPs or measured speedup. "
                        "All answer rows remain; KV column counts reflect optional pruning. Embedding, final norm and vocabulary "
                        "head still process the full sequence, equally in baseline and cached runs."}
        if any(p is not None for p in self.phases):
            result['refresh_phase_schedules']=[list(p) if p is not None else None for p in self.phases]
        result['refresh_counts_by_schedule_and_layer']=[
            [sum(e['layer_refresh'][l][i] for e in self.events) for l in range(len(self.blocks))]
            for i in range(len(self.schedules))]
        return result


def policy_context(llm, prepared, specification):
    plan = specification.get('pruning')
    if plan and any(s.get('refresh_phases') for s in specification['schedule']):
        raise ValueError('Phased cache schedules are evaluated without pruning')
    if plan and plan.get('engine') == 'compact_staged':
        if plan.get('temporal_cache'):
            from .cache_pruning import CachedProgressivePruning
            return CachedProgressivePruning(llm, prepared, specification)
        if any(s['refresh_every'] != 1 for s in specification['schedule']):
            raise ValueError('Compact staged pruning must be evaluated without cache reuse')
        from .staged_pruning import CompactStagedPruning
        if plan.get('progressive_global'):
            from .progressive_pruning import ProgressivePruning
            return ProgressivePruning(llm, prepared, plan)
        if plan.get('generation_trigger'):
            from .crossing_pruning import CrossingPruning
            return CrossingPruning(llm, prepared, plan)
        return CompactStagedPruning(llm, prepared, plan)
    return PrefixCache(llm, prepared, specification['schedule'], pruning=plan)
