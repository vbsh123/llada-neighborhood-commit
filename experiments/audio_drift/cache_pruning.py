"""Progressive audio pruning over the existing fixed-prefix KV/hidden cache.

Surviving cached rows remain KV columns. Removed rows leave both query updates
and KV attention at every layer on the next forward. Full backing banks retain
absolute coordinates, just as in PrefixCache; removed entries are never read.
"""
from copy import deepcopy
import math

import torch

from .compact import PrefixCache
from .progressive_pruning import ProgressivePruning, progressive_conditions


def combined_conditions(length, block_length, steps, acoustic_start=16,
                        semantic_start=44, interval=8, patience=2, threshold=.5,
                        fraction=.25, seed=1729):
    audio = progressive_conditions(length, block_length, steps, acoustic_start,
        semantic_start, interval, patience, threshold, fraction, seed)[0]
    result = []
    for a,p,s in ((16,8,16),(8,8,8)):
        schedule = [dict(group=g,refresh_every=r) for g,r in
                    [('acoustic',a),('prompt',p),('semantic',s)]]
        stem = f'freeze_triple_a{a}_p{p}_s{s}'
        result.append(dict(condition=stem,mode='freeze',schedule=schedule))
        plan = deepcopy(audio['pruning'])
        plan['temporal_cache'] = True
        result.append(dict(condition=stem+'__'+audio['condition'],mode='freeze',
                           schedule=deepcopy(schedule),pruning=plan))
    return result


class _AppliedRemovals:
    """PrefixCache's pruning interface, with decisions supplied externally."""
    def __init__(self):
        self.dropped = {}
        self.kept = None

    def select(self, layer, step, q, k, active):
        return self.kept

    def accounting(self):
        return {}


class CachedProgressivePruning(PrefixCache):
    def __init__(self,llm,prepared,specification):
        super().__init__(llm,prepared,specification['schedule'])
        self.controller = ProgressivePruning(llm,prepared,specification['pruning'])
        self.pruning = _AppliedRemovals()
        self.attention_originals = []
        self.embedding_handle = None

    def observe_input(self,x):
        c = self.controller
        c.observe_input(x)
        c.step = c.input_step
        if c.step != self.step+1:
            raise RuntimeError('Cache/pruning clocks diverged')
        removed = c.known_removals(0)
        if getattr(self,'last_removed',None) != removed:
            self.last_removed = removed
            removed_set = set(removed)
            self.key_ids = tuple(i for i in range(self.prepared.answer_start + c.spec['length'])
                                 if i not in removed_set)
            self.pruning.dropped = {l:removed for l in range(len(self.blocks))}
            self.pruning.kept = (torch.tensor(self.key_ids,device=x.device,dtype=torch.long)
                                 if removed else None)
        c.position_ids = self.key_ids
        self.observation_patterns = {}
        if c.monitoring:
            c.monitor_positions = torch.tensor(self.key_ids,device=x.device,dtype=torch.long)

    def __enter__(self):
        try:
            # PrefixCache installs the row-skipping block forwards. The observer
            # below sees their actual active queries and retained key columns.
            super().__enter__()
            self.embedding_handle = self.controller.llm.get_input_embeddings().register_forward_pre_hook(
                lambda module,args:self.observe_input(args[0]))
            for layer,block in enumerate(self.blocks):
                original = block._scaled_dot_product_attention
                existed = '_scaled_dot_product_attention' in block.__dict__
                self.attention_originals.append((block,existed,block.__dict__.get('_scaled_dot_product_attention')))
                def attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,
                              _layer=layer,_original=original):
                    c = self.controller
                    if c.monitoring:
                        if attn_mask is not None or dropout_p or is_causal:
                            raise ValueError('Requires dense bidirectional inference')
                        due = tuple(c.step==0 or bool(intervals[_layer] and c.step%intervals[_layer]==0)
                                    for _,intervals in self.schedules)
                        if due not in self.observation_patterns:
                            inactive = {p for (positions,_),refresh in zip(self.schedules,due)
                                        if not refresh for p in positions}
                            query_ids = [p for p in self.key_ids if p not in inactive]
                            queries = torch.tensor(query_ids,device=q.device,dtype=torch.long)
                            local = torch.searchsorted(queries,c.query_positions)
                            if not torch.equal(queries.index_select(0,local),c.query_positions):
                                raise RuntimeError('All eligible answer queries must remain active')
                            self.observation_patterns[due] = (len(query_ids),local)
                        query_count,local = self.observation_patterns[due]
                        if query_count!=q.shape[-2] or len(self.key_ids)!=k.shape[-2]:
                            raise RuntimeError('Cached progressive position map mismatch')
                        p = (q.index_select(-2,local).float()@k.float().transpose(-1,-2)
                             /math.sqrt(q.shape[-1])).softmax(-1)
                        scores = p.mean((0,1,2))
                        c.score_sum = scores if c.score_sum is None else c.score_sum+scores
                        c.score_layers += 1
                        if _layer==len(self.blocks)-1:
                            if c.score_layers!=len(self.blocks):raise RuntimeError('Incomplete pruning observation')
                            c.decide(c.score_sum/c.score_layers)
                    return _original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal)
                block._scaled_dot_product_attention = attention
        except Exception:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,*args):
        if self.embedding_handle:
            self.embedding_handle.remove()
            self.embedding_handle = None
        for block,existed,original in reversed(self.attention_originals):
            if existed:block._scaled_dot_product_attention = original
            else:delattr(block,'_scaled_dot_product_attention')
        self.attention_originals.clear()
        return super().__exit__(*args)

    def accounting(self):
        result = super().accounting()
        c = self.controller
        result['pruning'] = dict(specification=c.spec,selections=c.records,
            trigger_checks=c.checks,pending_unapplied=c.queued,
            activated_layers=len(self.blocks) if c.records else 0,
            note='Own-trajectory all-layer attention; previous-forward decisions apply before layer0. '
                 'Remaining prefix rows follow cache schedules. Dropped rows never return, including on refresh. '
                 'Full absolute-coordinate backing banks retained; removed KV columns excluded from attention.')
        return result
