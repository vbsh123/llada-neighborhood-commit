"""Repeated prefix pruning using previous-forward all-layer attention.

An observation completes before its decision takes effect on the next forward.
No current/future-layer information is used to modify an earlier layer in the
same forward. Retained rows are freshly computed, with original RoPE positions.
"""
from copy import deepcopy
import math

import torch

from .decode import schedule
from .staged_pruning import CompactStagedPruning


def progressive_conditions(length=512,block_length=32,steps=256,acoustic_start=16,
                           semantic_start=44,interval=8,patience=2,threshold=.5,
                           fraction=.25,seed=1729,layer_start=0):
    schedule(length,block_length,steps)
    if (not 1<=acoustic_start<=semantic_start<steps or interval<1 or patience<1
            or not math.isfinite(threshold) or threshold<=0
            or not math.isfinite(fraction) or not 0<fraction<=1
            or not isinstance(layer_start,int) or layer_start<0):
        raise ValueError('Invalid progressive pruning settings')
    stem=f'progressive_t{acoustic_start}x{semantic_start}_r{interval}_h{patience}_u{threshold:g}_f{fraction:g}'
    if layer_start:stem+=f'_d{layer_start}'
    result=[]
    for control in ('audio','early_text','late_text','generic'):
        plan=dict(engine='compact_staged',progressive_global=True,
            # Real stages are appended only when decisions are applied.
            stages=[dict(group='acoustic',layer=layer_start,start=10**12,fraction=0.)],
            generic=control=='generic',queries=1,seed=seed,scope='audio' if control=='audio' else control,
            selection='attention',start=acoustic_start,layer_start=layer_start,fraction=fraction,
            acoustic_start=acoustic_start,semantic_start=semantic_start,check_interval=interval,
            patience=patience,threshold_uniform=threshold,length=length,block_length=block_length,steps=steps,
            control=control)
        spec=dict(condition=stem+'_'+control,mode='freeze',schedule=[dict(group='prefix',refresh_every=1)],pruning=plan)
        if control!='audio':
            spec['control_source']=stem+'_audio';spec['paired_condition']=stem+'_audio'
        result.append(spec)
    return result


def resolve_progressive_control(spec,source):
    resolved=deepcopy(spec);plan=resolved['pruning'];plan['resolved_control']=True
    account=source['accounting'][0]['pruning']
    plan['monitor_steps']=[r['step'] for r in account['trigger_checks']]
    plan['source_events']=[dict(step=r['step'],group=r['group'],count=r['drop_budget']) for r in account['selections']]
    plan['control_note']='Reference-guided control: copy audio policy event times/counts, rank own positions. '
    if plan['control']=='generic':plan['control_note']+='Choose over all remaining prefix positions.'
    else:plan['control_note']+='Replace half the cumulative acoustic/semantic budget with lowest-attention prompt text; threshold waived for this forced text ablation. Natural prompt exhaustion is logged and remaining budget stays in the original group.'
    return resolved


class ProgressivePruning(CompactStagedPruning):
    def local_queries(self):
        return torch.searchsorted(self.monitor_positions,self.query_positions)

    def attention_scores(self, q, k):
        p=(q.index_select(-2,self.query_local).float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
        return p.mean((0,1,2))

    def __init__(self,llm,prepared,specification):
        super().__init__(llm,prepared,deepcopy(specification))
        self.llm=llm
        self.plan=schedule(self.spec['length'],self.spec['block_length'],self.spec['steps'])
        self.checks=[];self.queued=[];self.attention_originals=[];self.embedding_handle=None
        self.source_cumulative=dict(acoustic=0,semantic=0);self.replaced_cumulative=0
        if self.spec.get('resolved_control'):
            self.monitor_steps=set(self.spec['monitor_steps'])
        else:
            applies=set(range(self.spec['check_interval'],self.spec['steps'],self.spec['check_interval']))
            applies.update([self.spec['acoustic_start'],self.spec['semantic_start']])
            self.monitor_steps={t-1 for t in applies}

    def observe_input(self,x):
        step=self.step+1
        self.input_step=step
        # Register completed decisions before block work. Physical compaction
        # happens at the configured layer boundary, never in earlier blocks.
        for pick in self.queued:
            if pick['step']!=step:raise RuntimeError('Progressive decision applied at wrong forward')
            i=len(self.stages)
            layer=self.spec['layer_start']
            self.stages.append(dict(group=pick['group'],layer=layer,start=step,fraction=0.,drop_count=len(pick['dropped_positions'])))
            self.selected[i]=tuple(pick['dropped_positions'])
            self.records.append(dict(stage=i,layer=layer,applies_to_layers=list(range(layer,len(self.blocks))),
                drop_budget=len(pick['dropped_positions']),requested_fraction=self.spec['fraction'],**pick))
        self.queued=[]
        self.monitoring=step in self.monitor_steps
        if not self.monitoring:return
        block=self.plan[step][0]
        answer=x[0,self.prepared.prompt.shape[1]:]
        eligible=answer==self.prepared.mask_id
        eligible[(block+1)*self.spec['block_length']:]=False
        self.query_offsets=eligible.nonzero().flatten()
        if not len(self.query_offsets):raise ValueError('Progressive observation has no eligible query')
        self.query_positions=self.query_offsets+self.prepared.answer_start
        self.score_sum=None;self.relative_sum=None;self.score_layers=0
        if not hasattr(self,'streaks'):
            self.streaks=torch.zeros(self.prepared.answer_start,dtype=torch.long,device=x.device)

    def __enter__(self):
        try:
            super().__enter__()
            self.embedding_handle=self.llm.get_input_embeddings().register_forward_pre_hook(lambda module,args:self.observe_input(args[0]))
            for layer,block in enumerate(self.blocks):
                original=block._scaled_dot_product_attention
                existed='_scaled_dot_product_attention' in block.__dict__
                self.attention_originals.append((block,existed,block.__dict__.get('_scaled_dot_product_attention')))
                def attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_layer=layer,_original=original):
                    if getattr(self,'input_step',None)!=self.step:
                        raise RuntimeError('Progressive context requires forward_logits embedding input observation')
                    if self.monitoring:
                        with self.scope('prune.monitor_attention'):
                            if q.shape[1]!=k.shape[1] or attn_mask is not None or dropout_p or is_causal:
                                raise ValueError('Requires dense bidirectional equal-head attention')
                            if _layer==0 or self.spec['layer_start']:
                                self.monitor_positions=torch.tensor(self.position_ids,device=q.device,dtype=torch.long)
                                self.query_local=self.local_queries()
                            if len(self.monitor_positions)!=q.shape[-2]:raise RuntimeError('Position map mismatch')
                            scores=self.attention_scores(q,k)
                            if self.spec['layer_start']:
                                # Early layers retain all rows; deep layers have
                                # compact keys. Accumulate in absolute positions.
                                # Normalize against uniform at each layer before
                                # averaging, because key counts can differ.
                                if self.score_sum is None:
                                    self.score_sum=scores.new_zeros(self.length)
                                    self.relative_sum=scores.new_zeros(self.length)
                                self.score_sum.index_add_(0,self.monitor_positions,scores)
                                self.relative_sum.index_add_(0,self.monitor_positions,scores*len(self.monitor_positions))
                            else:
                                self.score_sum=scores if self.score_sum is None else self.score_sum+scores
                            self.score_layers+=1
                            if _layer==len(self.blocks)-1:
                                if self.score_layers!=len(self.blocks):raise RuntimeError('Incomplete all-layer scores')
                                if self.spec['layer_start']:
                                    # Only positions still retained in the deep
                                    # suffix can be chosen for further removal.
                                    self.decide(self.score_sum.index_select(0,self.monitor_positions)/self.score_layers,
                                        self.relative_sum.index_select(0,self.monitor_positions)/self.score_layers)
                                else:self.decide(self.score_sum/self.score_layers)
                    return _original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal)
                block._scaled_dot_product_attention=attention
        except Exception:
            self.__exit__(None,None,None);raise
        return self

    def decide(self,scores,relative_scores=None):
        active=self.monitor_positions[self.monitor_positions<self.prepared.answer_start]
        weights=scores[:len(active)]
        relative=weights*len(self.monitor_positions) if relative_scores is None else relative_scores[:len(active)]
        streak=torch.where(relative<self.spec['threshold_uniform'],self.streaks.index_select(0,active)+1,0)
        self.streaks.index_copy_(0,active,streak)
        eligible=streak>=self.spec['patience']
        apply=self.step+1;used=set();plans=[];replacement_details=[]
        active_ids=active.cpu().tolist()
        available=set(active_ids)
        groups={g:set(self.prepared.group_positions(g)) for g in ('acoustic','semantic','prompt')}
        if self.spec.get('resolved_control'):
            for event in self.spec['source_events']:
                if event['step']!=apply:continue
                group=event['group'];count=event['count']
                if self.spec['control']=='generic':plans.append((group,count,available,'generic'))
                else:
                    replace=((self.spec['control']=='early_text' and group=='acoustic') or
                             (self.spec['control']=='late_text' and group=='semantic'))
                    text_count=0
                    if replace:
                        self.source_cumulative[group]+=count
                        target=self.source_cumulative[group]//2
                        wanted=max(0,target-self.replaced_cumulative)
                        text_count=min(wanted,len((available&groups['prompt'])-used))
                        self.replaced_cumulative+=text_count
                        replacement_details.append(dict(group=group,requested_text_count=wanted,
                            allocated_text_count=text_count,prompt_available=len(available&groups['prompt'])))
                    if text_count:plans.append(('prompt',text_count,groups['prompt'],'forced_text'))
                    if count-text_count:plans.append((group,count-text_count,groups[group],'matched_audio'))
        else:
            for group,start in [('acoustic',self.spec['acoustic_start']),('semantic',self.spec['semantic_start'])]:
                if apply<start:continue
                local=[i for i,p in enumerate(active_ids) if p in groups[group]]
                local_tensor=torch.tensor(local,dtype=torch.long,device=scores.device)
                qualifying=local_tensor[eligible.index_select(0,local_tensor)]
                count=math.ceil(len(qualifying)*self.spec['fraction'])
                if count:plans.append((group,count,set(active.index_select(0,qualifying).cpu().tolist()),'threshold'))
        for group,count,pool,basis in plans:
            local=[i for i,p in enumerate(active_ids) if p in pool and p not in used]
            if count>len(local):raise ValueError('Matched control exhausted its available pool')
            local=torch.tensor(local,dtype=torch.long,device=scores.device)
            chosen=local.index_select(0,torch.argsort(weights.index_select(0,local),stable=True)[:count])
            positions=sorted(active.index_select(0,chosen).cpu().tolist());used.update(positions)
            self.queued.append(dict(group=group,step=apply,score_step=self.step,dropped_positions=positions,
                candidate_count=len(local),probe_positions=self.query_positions.cpu().tolist(),selection_basis=basis,
                removed_attention_mass=float(weights.index_select(0,chosen).sum()),
                dropped_by_group={g:len(set(positions)&groups[g]) for g in groups}))
        self.checks.append(dict(step=self.step,apply_step=apply,current_key_count=len(self.monitor_positions),
            query_positions=self.query_positions.cpu().tolist(),positions=active_ids,
            attention_mean=weights.cpu().tolist(),attention_relative_uniform=relative.cpu().tolist(),
            low_streak=streak.cpu().tolist(),threshold_uniform=self.spec['threshold_uniform'],
            eligible_by_group={g:sum(p in groups[g] and bool(e) for p,e in zip(active_ids,eligible.cpu().tolist())) for g in groups},
            replacement_details=replacement_details,
            planned_removals=[dict(group=p['group'],count=len(p['dropped_positions'])) for p in self.queued]))

    def __exit__(self,*args):
        if self.embedding_handle:self.embedding_handle.remove();self.embedding_handle=None
        for block,existed,original in reversed(self.attention_originals):
            if existed:block._scaled_dot_product_attention=original
            else:delattr(block,'_scaled_dot_product_attention')
        self.attention_originals.clear()
        return super().__exit__(*args)

    def accounting(self):
        result=super().accounting()
        result['pruning']['trigger_checks']=self.checks
        result['pruning']['pending_unapplied']=self.queued
        result['pruning']['budget_basis']=f"{100*self.spec['fraction']:g}% of currently threshold-qualified positions per group/check, ceil; controls match applied source events"
        result['pruning']['note']=f"Scores average all layers/heads/current eligible masked queries. Previous-forward decisions apply before layer{self.spec['layer_start']}; earlier layers keep all positions, no temporal KV cache. "\
            'Uniform attention normalization uses each layer\'s actual key count before averaging. Already removed deep positions cannot be selected again. Unapplied final decisions do not count as removals.'
        return result
