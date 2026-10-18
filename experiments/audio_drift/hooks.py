"""Observe post-RoPE KVs at the actual attention call, without replacing attention."""
from contextlib import AbstractContextManager
from types import MethodType

import torch


def blocks_of(llm):
    transformer = llm.model.transformer
    if "blocks" not in transformer or len(transformer.blocks) != llm.config.n_layers:
        raise ValueError("Expected pinned LLaDA with ungrouped transformer.blocks")
    return list(transformer.blocks)


class AttentionProbe(AbstractContextManager):
    """Frozen KVs are a sensitivity intervention, not optimized sparse execution.

    In freeze mode all queries/projections are still computed. The audio K/V
    columns passed to attention are replaced with post-RoPE values captured on
    the baseline anchor step. In remove mode those columns are absent for ALL
    queries, including indirect audio-to-text paths in later layers.
    """
    def __init__(self, llm, audio_positions, recorder=None, layers=None,
                 mode="observe", frozen=None, refresh_every=0, anchor_step=0, cache_device="cpu"):
        if mode not in {"observe", "capture", "freeze", "remove"}:
            raise ValueError(f"Unknown attention mode: {mode}")
        self.blocks = blocks_of(llm)
        if cache_device not in {"cpu", "model"}:
            raise ValueError("cache_device must be cpu or model")
        self.cache_device = cache_device
        self.device_positions = {}
        self.positions = torch.as_tensor(audio_positions, dtype=torch.long)
        if len(set(self.positions.tolist())) != len(self.positions) or (self.positions < 0).any():
            raise ValueError("Audio positions must be unique and nonnegative")
        if refresh_every < 0 or anchor_step < 0:
            raise ValueError("Refresh interval and anchor must be nonnegative")
        self.recorder = recorder
        self.layers = set(range(len(self.blocks))) if layers is None else set(layers)
        if not self.layers or not self.layers <= set(range(len(self.blocks))):
            raise ValueError("Invalid measurement layers")
        self.mode = mode
        # Each continuation owns its dictionary. Refreshes must never mutate
        # the baseline anchor bank shared by independent continuations.
        self.frozen = dict(frozen) if frozen is not None else {}
        self.refresh_every = refresh_every
        self.anchor_step = anchor_step
        self.forward_step = anchor_step
        self.last_refresh_step = anchor_step
        self.cache_events = []
        self.handles = []
        self.originals = []

    def __enter__(self):
        try:
            for layer, block in enumerate(self.blocks):
                original = block._scaled_dot_product_attention
                self.originals.append((block, original))

                def attention(_block, q, k, v, attn_mask=None, dropout_p=0.,
                              is_causal=False, _layer=layer, _original=original):
                    if q.shape[0] != 1 or k.shape[-2] != q.shape[-2]:
                        raise ValueError("Audio probe requires batch 1, no past KV cache")
                    if attn_mask is not None or dropout_p or is_causal:
                        raise ValueError("Expected unmasked bidirectional eval attention")
                    if _layer == 0:
                        self.forward_step += 1
                        if self.mode == "freeze":
                            refresh = bool(self.refresh_every and
                                           (self.forward_step - self.anchor_step) % self.refresh_every == 0)
                            if refresh:
                                self.last_refresh_step = self.forward_step
                            self.cache_events.append({"step": self.forward_step,
                                "refreshed": refresh,
                                "cache_age": self.forward_step - self.last_refresh_step})
                    if k.device not in self.device_positions:
                        self.device_positions[k.device] = self.positions.to(k.device)
                    positions = self.device_positions[k.device]
                    def store(t):
                        # index_select owns its storage; retaining it never aliases
                        # live attention tensors or mutates an anchor bank.
                        selected = t.index_select(-2, positions).detach()
                        return selected.cpu() if self.cache_device == "cpu" else selected
                    if self.recorder is not None and _layer in self.layers:
                        # [1, heads, sequence, head_dim] -> [sequence, heads*head_dim]
                        for name, tensor in (("key", k), ("value", v)):
                            self.recorder.observe(_layer, name, tensor[0].transpose(0, 1).flatten(1))
                    if self.mode == "capture":
                        self.frozen[_layer] = tuple(store(t) for t in (k, v))
                    elif self.mode == "freeze":
                        if self.cache_events[-1]["refreshed"]:
                            # Refresh from THIS branch's current state, never
                            # from an oracle baseline future representation.
                            self.frozen[_layer] = tuple(store(t) for t in (k, v))
                        else:
                            old_k, old_v = self.frozen[_layer]
                            k, v = k.clone(), v.clone()
                            k.index_copy_(-2, positions, old_k.to(k.device, k.dtype))
                            v.index_copy_(-2, positions, old_v.to(v.device, v.dtype))
                    elif self.mode == "remove":
                        keep = torch.ones(k.shape[-2], dtype=torch.bool, device=k.device)
                        keep[positions] = False
                        k, v = k[:, :, keep, :], v[:, :, keep, :]
                    return _original(q, k, v, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal)

                block._scaled_dot_product_attention = MethodType(attention, block)
                if self.recorder is not None and layer in self.layers:
                    def hidden(_module, _inputs, output, _layer=layer):
                        self.recorder.observe(_layer, "hidden", output[0][0])
                    self.handles.append(block.register_forward_hook(hidden))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        for block, original in self.originals:
            block._scaled_dot_product_attention = original
        self.handles.clear()
        self.originals.clear()
