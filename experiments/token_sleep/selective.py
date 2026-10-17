"""Real row-selective execution using pinned LLaDA modules and unchanged weights."""
from contextlib import contextmanager
import types

from .simulation import checked_blocks


class SelectiveForward:
    """Only awake rows enter QKV/attention/FFN; all K/V positions remain context.

    The block's original forward executes unchanged on a narrower hidden tensor.
    Only its attention method is temporarily replaced to handle original-position
    RoPE and rectangular attention. Cache is per sample, post-RoPE, per layer.
    """

    def __init__(self, model, *, blocks=None):
        self.model = model
        self.blocks = checked_blocks(model) if blocks is None else list(blocks)
        self.transformer = model.model.transformer
        self.config = self.blocks[0].config
        if model.training or self.config.alibi or not self.config.rope:
            raise ValueError('Requires eval mode and the pinned RoPE/non-ALiBi layout')
        self.kv = [None for _ in self.blocks]
        self.sequence_length = None
        self.last_active_count = self.last_head_count = 0

    @contextmanager
    def attention_methods(self, active, sequence_length):
        import torch
        originals = []
        # Pinned LLaDA layers share rotary buffers. Gather the same positions
        # once per buffer/dtype per forward, rather than once per layer.
        rotary_tables = {}
        try:
            for index, block in enumerate(self.blocks):
                originals.append((block, block.attention))

                def attention(this, q, k, v, attention_bias=None, layer_past=None, use_cache=False,
                              _index=index):
                    if attention_bias is not None or layer_past is not None or use_cache:
                        raise ValueError('Padding/bias/AR cache unsupported by this batch-one probe')
                    batch, count, width = q.shape
                    dtype = k.dtype
                    if this.q_norm is not None and this.k_norm is not None:
                        q = this.q_norm(q).to(dtype=dtype)
                        k = this.k_norm(k).to(dtype=dtype)
                    head_dim = width//this.config.n_heads
                    q = q.view(batch,count,this.config.n_heads,head_dim).transpose(1,2)
                    k = k.view(batch,count,this.config.effective_n_kv_heads,head_dim).transpose(1,2)
                    v = v.view(batch,count,this.config.effective_n_kv_heads,head_dim).transpose(1,2)
                    if count == sequence_length:
                        # Native full-sequence RoPE: identity position gathers
                        # are unnecessary on every global refresh.
                        q_rot, k_rot = this.rotary_emb(q,k)
                    else:
                        q_work, k_work = (q.float(), k.float()) if this.config.rope_full_precision else (q,k)
                        with torch.autocast(q.device.type, enabled=False):
                            source_sin, source_cos = this.rotary_emb.get_rotary_embedding(sequence_length, q.device)
                            key = (source_sin.data_ptr(), source_cos.data_ptr(),
                                   tuple(source_sin.shape), tuple(source_sin.stride()),
                                   tuple(source_cos.shape), tuple(source_cos.stride()),
                                   q.device, q_work.dtype)
                            if key not in rotary_tables:
                                sin = source_sin.type_as(q_work).index_select(2,active)
                                cos = source_cos.type_as(q_work).index_select(2,active)
                                # Retain source buffers too: prevent allocator
                                # pointer reuse from aliasing distinct tables.
                                rotary_tables[key] = (sin,cos,source_sin,source_cos)
                            sin,cos,_,_ = rotary_tables[key]
                            q_rot = this.rotary_emb.apply_rotary_pos_emb(sin,cos,q_work).type_as(q)
                            k_rot = this.rotary_emb.apply_rotary_pos_emb(sin,cos,k_work).type_as(k)
                    if count == sequence_length:
                        # Already in original sorted order: no scatter or clone.
                        self.kv[_index] = (k_rot,v)
                    else:
                        if self.kv[_index] is None:
                            raise ValueError('First forward of each sample must initialize every position')
                        old_k, old_v = self.kv[_index]
                        old_k.index_copy_(2,active,k_rot)
                        old_v.index_copy_(2,active,v)
                    full_k, full_v = self.kv[_index]
                    attended = this._scaled_dot_product_attention(q_rot,full_k,full_v,
                        attn_mask=None,dropout_p=0.,is_causal=False)
                    attended = attended.transpose(1,2).contiguous().view(batch,count,width)
                    return this.attn_out(attended), None

                block.attention = types.MethodType(attention,block)
            yield
        finally:
            for block, original in originals:
                block.attention = original

    def __call__(self, input_ids, sleepers, prediction_positions):
        """All indices are absolute sequence indices; output order matches predictions."""
        import torch
        from torch.nn import functional as F
        if input_ids.shape[0] != 1 or torch.is_grad_enabled():
            raise ValueError('Requires batch one and inference_mode/no_grad')
        n = input_ids.shape[1]
        if self.sequence_length not in (None,n):
            raise ValueError('Create a new SelectiveForward for each sequence')
        self.sequence_length = n
        sleeping = set(sleepers)
        if any(p < 0 or p >= n for p in sleeping) or len(sleeping) != len(sleepers):
            raise ValueError('Invalid/duplicate sleeper indices')
        active_list = [p for p in range(n) if p not in sleeping]
        if not prediction_positions or len(set(prediction_positions)) != len(prediction_positions):
            raise ValueError('Predictions must be nonempty and unique')
        if any(p < 0 or p >= n or p in sleeping for p in prediction_positions):
            raise ValueError('Cannot predict a sleeper or an out-of-range position')
        if any(cache is None for cache in self.kv) and sleeping:
            raise ValueError('First forward must be fully awake')
        if sleeping:
            active = torch.tensor(active_list, device=input_ids.device, dtype=torch.long)
            lookup = {p:i for i,p in enumerate(active_list)}
            local_list = [lookup[p] for p in prediction_positions]
            selected_ids = input_ids.index_select(1,active)
        else:
            active = None
            local_list = prediction_positions
            selected_ids = input_ids
        local = torch.tensor(local_list, device=input_ids.device, dtype=torch.long)
        hidden = self.transformer.wte(selected_ids)
        if self.config.input_emb_norm:
            hidden = hidden*(self.config.d_model**.5)
        hidden = self.transformer.emb_drop(hidden)
        with self.attention_methods(active,n):
            for block in self.blocks:
                hidden,_ = block(hidden,use_cache=False)
        hidden = self.transformer.ln_f(hidden.index_select(1,local))
        if self.config.weight_tying:
            logits = F.linear(hidden,self.transformer.wte.weight,None)
        else:
            logits = self.transformer.ff_out(hidden)
        if self.config.scale_logits:
            logits.mul_(1/self.config.d_model**.5)
        self.last_active_count = len(active_list)
        self.last_head_count = len(prediction_positions)
        return logits
