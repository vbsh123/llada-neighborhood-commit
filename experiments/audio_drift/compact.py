"""Skip cached fixed-prefix rows through QKV, attention queries and feed-forward.

Inference-only adapter for the pinned, ungrouped LLaDALlamaBlock. All answer
rows remain active. Absolute RoPE positions and the complete key sequence are
preserved. No upstream source changes or token pruning.
"""
from contextlib import AbstractContextManager

import torch

from .hooks import blocks_of
from .protocol import layer_intervals


class PrefixCache(AbstractContextManager):
    def __init__(self, llm, prepared, schedules):
        if llm.training:
            raise ValueError("PrefixCache requires eval mode")
        self.blocks = blocks_of(llm)
        self.prepared = prepared
        self.schedules = []
        claimed = set()
        for spec in schedules:
            positions = prepared.group_positions(spec["group"])
            if not positions or claimed.intersection(positions) or spec["refresh_every"] < 0:
                raise ValueError("Cache schedules must select nonempty disjoint fixed-prefix groups")
            claimed.update(positions)
            self.schedules.append((positions, layer_intervals(spec, len(self.blocks))))
        self.step = -1
        self.patterns = {}
        self.banks = {}
        self.events = []
        self.originals = []

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
                                            "layer_computed_rows": [], "layer_refresh": []})
                    # Each layer has its own clock and banks. Flat schedules
                    # expand to identical intervals at every layer.
                    due = tuple(self.step == 0 or bool(intervals[_layer] and
                                self.step % intervals[_layer] == 0) for _, intervals in self.schedules)
                    key = (x.shape[1], x.device, due)
                    if key not in self.patterns:
                        inactive = {p for (positions, _), refresh in zip(self.schedules, due)
                                    if not refresh for p in positions}
                        selected = [p for p in range(x.shape[1]) if p not in inactive]
                        self.patterns[key] = torch.tensor(selected, dtype=torch.long, device=x.device)
                    active = self.patterns[key]
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
                    attended = _block._scaled_dot_product_attention(q_rot.type_as(q), full_k, full_v,
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
        if any(len(e['layer_computed_rows']) != len(self.blocks) for e in self.events):
            raise RuntimeError("Incomplete transformer forwards in cache accounting")
        full = sum(e["total_rows"] * len(self.blocks) for e in self.events)
        used = sum(sum(e['layer_computed_rows']) for e in self.events)
        return {"forwards": len(self.events), "computed_token_layers": used,
                "dense_token_layers_at_same_forward_count": full,
                "computed_rows_by_layer": [sum(e['layer_computed_rows'][i] for e in self.events)
                                           for i in range(len(self.blocks))],
                "effective_schedules": [{"positions": len(positions), "refresh_by_layer": list(intervals)}
                                        for positions, intervals in self.schedules],
                "skipped_token_update_fraction": 1 - used / max(1, full),
                "note": "Counts skipped QKV, attention query/output and MLP rows; not total FLOPs or measured speedup. "
                        "All answer rows and all KV columns remain present. Embedding, final norm and vocabulary "
                        "head still process the full sequence, equally in baseline and cached runs."}
