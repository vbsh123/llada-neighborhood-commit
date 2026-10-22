"""Online audio-versus-prompt crossing with progressive compact pruning.

The generic control uses the audio policy's event times and token budgets, but
selects its own lowest-attention prefix positions. No reference answer or future
baseline completion length is used by the audio policy.
"""
from copy import deepcopy
import math

import torch

from .staged_pruning import CompactStagedPruning


def crossing_conditions(fractions=(.25,.5,.75), layer=25, start=8, interval=8,
                        patience=2, queries=64, seed=1729):
    if (not fractions or any(not math.isfinite(f) or not 0<f<=1 for f in fractions)
            or list(fractions)!=sorted(set(fractions)) or layer<0 or start<1
            or interval<1 or patience<1 or queries<1):
        raise ValueError('Invalid crossing pruning configuration')
    stages=[]
    for group in ('acoustic','semantic'):
        previous=0.
        for f in fractions:
            stages.append(dict(group=group,layer=layer,start=start,fraction=f,previous_fraction=previous))
            previous=f
    tag='x'.join(format(100*f,'.8g').replace('.','p') for f in fractions)
    stem=f'crossing_d{layer}_t{start}_r{interval}_h{patience}_f{tag}'
    result=[]
    for generic in (False,True):
        name=stem+('_generic' if generic else '_audio')
        plan=dict(engine='compact_staged',generation_trigger=True,stages=stages,generic=generic,
                  queries=queries,seed=seed,scope='prefix' if generic else 'audio',selection='attention',
                  start=start,layer_start=layer,fraction=fractions[-1],check_interval=interval,patience=patience)
        spec=dict(condition=name,paired_condition=stem+('_audio' if generic else '_generic'),
                  mode='freeze',schedule=[dict(group='prefix',refresh_every=1)],pruning=plan)
        if generic:spec['control_source']=stem+'_audio'
        result.append(spec)
    return result


def dependency_order(specs):
    """Keep the shuffled order where possible; source must precede control."""
    remaining=list(specs);ordered=[];seen=set()
    while remaining:
        ready=next((s for s in remaining if not s.get('control_source') or s['control_source'] in seen),None)
        if ready is None:raise ValueError('Crossing control requires its source policy in the same benchmark')
        ordered.append(ready);seen.add(ready['condition']);remaining.remove(ready)
    return ordered


def resolve_control(spec, completed):
    if not spec.get('control_source'):return spec
    source=next((p for p in completed if p['condition']==spec['control_source']),None)
    if source is None:raise ValueError('Missing crossing source measurements')
    if spec['pruning'].get('progressive_global'):
        from .progressive_pruning import resolve_progressive_control
        return resolve_progressive_control(spec,source)
    account=source['accounting'][0]['pruning']
    resolved=deepcopy(spec);plan=resolved['pruning']
    plan['resolved_control']=True
    plan['stages']=[dict(group=p['group'],layer=p['layer'],start=p['step'],
                         fraction=p['requested_fraction'],drop_count=p['drop_budget'])
                    for p in account['selections']]
    # A non-triggered source must produce an equally non-pruning control.
    if not plan['stages']:
        plan['stages']=[dict(group='acoustic',layer=plan['layer_start'],start=10**12,fraction=0.)]
    plan['monitor_steps']=[c['step'] for c in account['trigger_checks']]
    plan['control_note']='Event times/counts copied from paired autonomous audio run (or paired replay for replay). Positions ranked independently over full prefix; not a standalone adaptive control.'
    return resolved


class CrossingPruning(CompactStagedPruning):
    def __init__(self,llm,prepared,specification):
        super().__init__(llm,prepared,specification)
        self.checks=[];self.streaks=dict(acoustic=0,semantic=0)

    def should_score(self,layer,pending):
        if layer!=self.spec['layer_start']:return False
        if self.spec.get('resolved_control'):
            return bool(pending) or self.step in self.spec['monitor_steps']
        return (self.step>=self.spec['start'] and
                (self.step-self.spec['start'])%self.spec['check_interval']==0 and
                len(self.selected)<len(self.stages))

    def choose_pending(self,scores,positions,pending):
        masses=[]
        available=set(self.position_ids)
        for group in ('acoustic','semantic','prompt'):
            ids=sorted(available.intersection(self.prepared.group_positions(group)))
            local=torch.searchsorted(positions,torch.tensor(ids,dtype=torch.long,device=positions.device))
            masses.append(scores.index_select(0,local).sum())
        acoustic,semantic,prompt=torch.stack(masses).detach().cpu().tolist()
        record=dict(step=self.step,layer=self.spec['layer_start'],acoustic_mass=acoustic,
                    semantic_mass=semantic,prompt_mass=prompt,source='own_current_surviving_positions')
        if self.spec.get('resolved_control'):
            record['forced_control_events']=pending
            self.checks.append(record)
            return pending
        chosen=[]
        for group,mass in [('acoustic',acoustic),('semantic',semantic)]:
            next_stage=next((i for i,s in enumerate(self.stages) if s['group']==group and i not in self.selected),None)
            if next_stage is None:continue
            self.streaks[group]=self.streaks[group]+1 if mass<prompt else 0
            if self.streaks[group]>=self.spec['patience']:
                chosen.append(next_stage);self.streaks[group]=0
        record['selected_stages']=chosen;record['streaks_after_check']=dict(self.streaks)
        self.checks.append(record)
        return chosen

    def accounting(self):
        result=super().accounting()
        result['pruning']['trigger_checks']=self.checks
        result['pruning']['trigger_note']='Total remaining group attention versus remaining prompt-text mass at the first pruned layer. Two groups trigger independently; a new patience window is required per tier. Removal can reinforce later crossings. No future answer-length signal.'
        return result
