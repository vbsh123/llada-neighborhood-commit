"""All-layer physical pruning chosen by global incoming attention importance.

Observe every surviving query row/head/layer. Apply decisions next forward.
No current-block query restriction, temporal KV reuse, deep-only option or cap.
"""
from contextlib import AbstractContextManager
import math

import torch


class GlobalPruning(AbstractContextManager):
    def __init__(self,llm,prepared,*,control='audio',source=None,threshold=.5,
                 fraction=.125,interval=8,patience=2,starts=(16,44),query_chunk=32):
        self.blocks=list(llm.model.transformer.blocks)
        if (llm.training or control not in ('audio','generic') or (control=='generic')!=(source is not None)
                or not math.isfinite(threshold) or threshold<=0 or not math.isfinite(fraction) or not 0<fraction<=1
                or min(interval,patience,query_chunk)<1 or not 1<=starts[0]<=starts[1]<256):
            raise ValueError('Invalid global pruning settings')
        self.prepared,self.control,self.source=prepared,control,source
        self.threshold,self.fraction,self.interval,self.patience=threshold,fraction,interval,patience
        self.starts,self.query_chunk=tuple(starts),query_chunk
        self.groups={g:set(prepared.group_positions(g)) for g in ('acoustic','prompt','semantic')}
        applies=set(range(interval,256,interval))|set(starts)
        self.monitor_steps=({c['step'] for c in source['checks']} if source is not None else {s-1 for s in applies})
        self.step,self.length=-1,None
        self.dropped,self.pending=set(),[]
        self.checks,self.selections,self.events=[],[],[]
        self.originals,self.attention_originals=[],[]
        self.patterns={}

    def begin(self,x):
        self.step+=1
        if self.length is None:
            self.length=x.shape[1]
            self.streak=torch.zeros(self.prepared.answer_start,device=x.device,dtype=torch.long)
        if x.shape[1]!=self.length:
            raise ValueError('Full canvas required at each forward entry')
        for event in self.pending:
            if event['step']!=self.step:
                raise RuntimeError('Pruning decision applied on the wrong forward')
            self.dropped.update(event['dropped_positions'])
            self.selections.append(event)
        self.pending=[]
        pattern=tuple(sorted(self.dropped))
        if pattern not in self.patterns:
            positions=tuple(p for p in range(self.length) if p not in self.dropped)
            self.patterns[pattern]=(positions,torch.tensor(positions,device=x.device,dtype=torch.long))
        self.position_ids,self.positions=self.patterns[pattern]
        self.monitoring=self.step in self.monitor_steps
        self.score_sum=None
        self.score_layers=0
        self.events.append(dict(step=self.step,computed_rows=len(self.positions),layers=0))

    def observe(self,q,k,layer):
        if q.shape[-2]!=len(self.positions) or k.shape[-2]!=len(self.positions) or q.shape[1]!=k.shape[1]:
            raise ValueError('Global scoring requires equal-head self attention and valid row coordinates')
        total=k.new_zeros(k.shape[-2],dtype=torch.float32)
        keys=k.float().transpose(-1,-2)
        for start in range(0,q.shape[-2],self.query_chunk):
            probability=(q[:,:,start:start+self.query_chunk].float()@keys/math.sqrt(q.shape[-1])).softmax(-1)
            total+=probability.sum((0,1,2))
        mean=total/(q.shape[0]*q.shape[1]*q.shape[-2])
        self.score_sum=mean if self.score_sum is None else self.score_sum+mean
        self.score_layers+=1
        if layer==len(self.blocks)-1:
            if self.score_layers!=len(self.blocks):
                raise RuntimeError('Incomplete global-attention observation')
            self.decide(self.score_sum/self.score_layers)

    def decide(self,scores):
        active=[p for p in self.position_ids if p<self.prepared.answer_start]
        positions=self.positions[:len(active)]
        weights=scores[:len(active)]
        relative=weights*len(self.positions)
        streak=torch.where(relative<self.threshold,self.streak.index_select(0,positions)+1,0)
        self.streak.index_copy_(0,positions,streak)
        qualified=streak>=self.patience
        apply=self.step+1
        plans=[]
        if self.source is None:
            flags=qualified.cpu().tolist()
            for group,start in zip(('acoustic','semantic'),self.starts):
                if apply>=start:
                    pool=[i for i,p in enumerate(active) if p in self.groups[group] and flags[i]]
                    count=math.ceil(len(pool)*self.fraction)
                    if count:
                        plans.append((group,count,pool))
        else:
            for event in self.source['selections']:
                if event['step']==apply:
                    plans.append((event['group'],event['drop_budget'],list(range(len(active)))))
        used=set()
        for group,count,pool in plans:
            pool=[i for i in pool if active[i] not in used]
            if len(pool)<count:
                raise ValueError('Generic control exhausted its candidate pool')
            local=torch.tensor(pool,device=weights.device,dtype=torch.long)
            pick=local[torch.argsort(weights.index_select(0,local),stable=True)[:count]]
            dropped=sorted(positions.index_select(0,pick).cpu().tolist())
            used.update(dropped)
            self.pending.append(dict(step=apply,score_step=self.step,group=group,drop_budget=count,
                dropped_positions=dropped,removed_attention_mass=float(weights.index_select(0,pick).sum()),
                dropped_by_group={g:len(set(dropped)&ids) for g,ids in self.groups.items()}))
        self.checks.append(dict(step=self.step,apply_step=apply,current_key_count=len(self.positions),
            query_count_per_layer=len(self.positions),query_scope='all_surviving_positions',
            eligible_by_group={g:sum(p in ids and bool(flag) for p,flag in zip(active,qualified.cpu().tolist()))
                               for g,ids in self.groups.items()},
            attention_mass_by_group={g:float(weights[[i for i,p in enumerate(active) if p in ids]].sum())
                                     for g,ids in self.groups.items()},
            planned_removals=[dict(group=e['group'],count=e['drop_budget']) for e in self.pending]))

    def __enter__(self):
        try:
            for layer,block in enumerate(self.blocks):
                if (type(block).__name__!='LLaDALlamaBlock' or block._activation_checkpoint_fn is not None
                        or not block.config.rope or block.config.n_heads!=block.config.effective_n_kv_heads):
                    raise ValueError('Requires pinned ungrouped rotary equal-head LLaDA')
                original_attention=block._scaled_dot_product_attention
                self.attention_originals.append((block,'_scaled_dot_product_attention' in block.__dict__,
                                                block.__dict__.get('_scaled_dot_product_attention')))
                def attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_layer=layer,_original=original_attention):
                    if attn_mask is not None or dropout_p or is_causal:
                        raise ValueError('Requires unmasked bidirectional inference')
                    if self.monitoring:
                        self.observe(q,k,_layer)
                    return _original(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False)
                block._scaled_dot_product_attention=attention
                original=block.forward
                self.originals.append((block,'forward' in block.__dict__,block.__dict__.get('forward')))
                def forward(x,attention_bias=None,layer_past=None,use_cache=False,replace_position=None,
                            _layer=layer,_block=block,_original=original):
                    if x.shape[0]!=1 or attention_bias is not None or layer_past is not None or use_cache or replace_position is not None:
                        raise ValueError('Requires batch-one bidirectional no-past inference')
                    if _layer==0:
                        self.begin(x)
                        if self.dropped:
                            x=x.index_select(1,self.positions)
                    self.events[-1]['layers']+=1
                    if not self.dropped:
                        return _original(x)
                    norm=_block.attn_norm(x)
                    q,k,v=_block.q_proj(norm),_block.k_proj(norm),_block.v_proj(norm)
                    if _block.q_norm is not None and _block.k_norm is not None:
                        q,k=_block.q_norm(q).to(k.dtype),_block.k_norm(k).to(k.dtype)
                    batch,rows,width=q.shape
                    heads=_block.config.n_heads
                    q=q.view(batch,rows,heads,width//heads).transpose(1,2)
                    k=k.view(batch,rows,heads,width//heads).transpose(1,2)
                    v=v.view(batch,rows,heads,width//heads).transpose(1,2)
                    qr,kr=(q.float(),k.float()) if _block.config.rope_full_precision else (q,k)
                    rotary=_block.rotary_emb
                    with torch.autocast(q.device.type,enabled=False):
                        sin,cos=rotary.get_rotary_embedding(self.length,q.device)
                        sin=sin.type_as(qr).index_select(-2,self.positions)
                        cos=cos.type_as(qr).index_select(-2,self.positions)
                        qr=rotary.apply_rotary_pos_emb(sin,cos,qr)
                        kr=rotary.apply_rotary_pos_emb(sin,cos,kr)
                    attended=_block._scaled_dot_product_attention(qr.type_as(q),kr.type_as(k),v,
                        attn_mask=None,dropout_p=0.,is_causal=False)
                    attended=attended.transpose(1,2).contiguous().view(batch,rows,width)
                    x=x+_block.dropout(_block.attn_out(attended))
                    norm=_block.ff_norm(x)
                    x=x+_block.dropout(_block.ff_out(_block.act(_block.ff_proj(norm))*_block.up_proj(norm)))
                    if _layer==len(self.blocks)-1:
                        full=x.new_zeros((batch,self.length,width))
                        full.index_copy_(1,self.positions,x)
                        x=full
                    return x,None
                block.forward=forward
        except Exception:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,*args):
        for block,existed,original in reversed(self.originals):
            if existed: block.forward=original
            else: delattr(block,'forward')
        for block,existed,original in reversed(self.attention_originals):
            if existed: block._scaled_dot_product_attention=original
            else: delattr(block,'_scaled_dot_product_attention')
        self.originals.clear()
        self.attention_originals.clear()

    def accounting(self):
        if any(e['layers']!=len(self.blocks) for e in self.events):
            raise RuntimeError('Incomplete pruning forward')
        used=sum(e['computed_rows'] for e in self.events)*len(self.blocks)
        dense=len(self.events)*(self.length or 0)*len(self.blocks)
        return dict(forwards=len(self.events),computed_token_layers=used,dense_token_layers=dense,
            skipped_token_update_fraction=1-used/max(1,dense),
            pruning=dict(control=self.control,query_scope='all_surviving_positions',threshold_uniform=self.threshold,
                fraction=self.fraction,starts=self.starts,interval=self.interval,patience=self.patience,
                checks=self.checks,selections=self.selections,pending_unapplied=self.pending,
                note='Exact attention averaged over all surviving queries/heads/layers, including self attention. '
                     'Observe on preceding forward, then physically remove rows and KV columns in all layers. '
                     'No cache, token cap, query subsampling or current-answer-block restriction. '
                     'Generic control copies audio event times/counts and ranks its own entire retained prefix.'))
