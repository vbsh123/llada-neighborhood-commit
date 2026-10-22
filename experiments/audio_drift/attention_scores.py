"""Online QK summaries from eligible masked queries, without changing attention."""
from contextlib import AbstractContextManager
import math

import torch

from .hooks import blocks_of


class OnlineAttentionScores(AbstractContextManager):
    def __init__(self,llm,prepared,layers=None,special_ids=()):
        self.blocks=blocks_of(llm);self.prepared=prepared
        self.layers=set(range(len(self.blocks))) if layers is None else set(layers)
        if not self.layers or not self.layers<=set(range(len(self.blocks))):
            raise ValueError('Invalid score layers')
        self.special_ids=set(special_ids)|{prepared.mask_id}
        self.originals=[]

    def begin(self,x,block,block_length,eos_id=None):
        answer=x[0,self.prepared.prompt.shape[1]:]
        eligible=(answer==self.prepared.mask_id)
        eligible[(block+1)*block_length:]=False
        self.query_offsets=eligible.nonzero().flatten()
        if not len(self.query_offsets):raise ValueError('No eligible queries')
        self.queries=self.query_offsets+self.prepared.answer_start
        revealed=torch.ones_like(answer,dtype=torch.bool)
        for special in self.special_ids:revealed &= answer!=special
        if eos_id is not None:
            eos=(answer==eos_id).nonzero().flatten()
            if len(eos):revealed[int(eos[0]):]=False
        self.revealed=revealed.nonzero().flatten()+self.prepared.answer_start
        self.key_scores=None;self.query_mass=None;self.observed=0;self.layer_mass=[]

    def __enter__(self):
        try:
            for layer,block in enumerate(self.blocks):
                if layer not in self.layers:continue
                original=block._scaled_dot_product_attention
                existed='_scaled_dot_product_attention' in block.__dict__
                self.originals.append((block,existed,block.__dict__.get('_scaled_dot_product_attention')))
                def attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_original=original):
                    if attn_mask is not None or dropout_p or is_causal or q.shape[0]!=1 or k.shape[-2]!=q.shape[-2]:
                        raise ValueError('Online scores require dense bidirectional batch-one attention')
                    # DIFFA's pinned LLaDA uses equal query/KV head counts.
                    if q.shape[1]!=k.shape[1]:raise ValueError('GQA is not supported by this diagnostic')
                    probability=(q.index_select(-2,self.queries).float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
                    weights=probability.mean((0,1))  # [eligible queries, keys]
                    key=weights[:,:self.prepared.answer_start].mean(0)
                    groups=[self.prepared.group_positions(g) for g in ('semantic','prompt','acoustic')]
                    masses=[weights.index_select(-1,torch.tensor(p,dtype=torch.long,device=q.device)).sum(-1) for p in groups]
                    masses.append(weights.index_select(-1,self.revealed).sum(-1))
                    mass=torch.stack(masses,-1)
                    self.layer_mass.append(mass.mean(0))
                    self.key_scores=key if self.key_scores is None else self.key_scores+key
                    self.query_mass=mass if self.query_mass is None else self.query_mass+mass
                    self.observed+=1
                    return _original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal)
                block._scaled_dot_product_attention=attention
        except Exception:
            self.__exit__(None,None,None);raise
        return self

    def summaries(self):
        if self.observed!=len(self.layers):raise RuntimeError('Incomplete attention collection')
        return self.key_scores/self.observed,self.query_mass/self.observed

    def __exit__(self,*args):
        for block,existed,original in reversed(self.originals):
            if existed:block._scaled_dot_product_attention=original
            else:delattr(block,'_scaled_dot_product_attention')
        self.originals.clear()
