"""Attention-dependent commitment priority; probabilities/logits stay unchanged."""
import math
import time

import torch

from .attention_scores import OnlineAttentionScores
from .decode import decode


class BalanceRamp:
    """Smoothed mass balance with an observed post-start semantic-to-text crossing."""
    def __init__(self,start=40,smoothing=.2):
        if start<0 or not 0<smoothing<=1:raise ValueError('Invalid balance ramp')
        self.start=start;self.smoothing=smoothing
        self.ema=None;self.anchor=None;self.cutoff=None;self.left=1.;self.right=0.;self.previous=None

    def update(self,step,semantic,text):
        if step<self.start:return dict(semantic_weight=0.,text_weight=0.,phase='before_start',cutoff=None)
        if self.ema is None:self.ema=(semantic,text)
        else:self.ema=tuple((1-self.smoothing)*old+self.smoothing*new for old,new in zip(self.ema,(semantic,text)))
        s,t=self.ema;margin=(s-t)/max(s+t,1e-12)
        rates=dict(semantic_change_per_forward=None,text_change_per_forward=None,margin_change_per_forward=None)
        if self.previous is not None:
            old_step,old_s,old_t,old_margin=self.previous
            elapsed=step-old_step
            if elapsed<=0:raise ValueError('Balance updates must advance forward index')
            rates=dict(semantic_change_per_forward=(s-old_s)/elapsed,text_change_per_forward=(t-old_t)/elapsed,
                       margin_change_per_forward=(margin-old_margin)/elapsed)
        self.previous=(step,s,t,margin)
        if self.anchor is None:
            if margin>0:self.anchor=margin
            else:return dict(semantic_weight=0.,text_weight=0.,phase='waiting_for_semantic_dominance',cutoff=None,
                             semantic_ema=s,text_ema=t,semantic_margin=margin,**rates)
        if self.cutoff is None:
            self.left=min(self.left,max(0.,margin/self.anchor))
            if margin<=0:self.cutoff=step;self.left=0.
        if self.cutoff is not None:self.right=max(self.right,max(0.,-margin))
        return dict(semantic_weight=self.left if self.cutoff is None else 0.,text_weight=self.right,
                    phase='semantic' if self.cutoff is None else 'text',cutoff=self.cutoff,
                    semantic_ema=s,text_ema=t,semantic_margin=margin,anchor_margin=self.anchor,**rates)


class FixedPhaseRamp:
    """Semantic from zero to midpoint; fixed-prompt ramp after midpoint."""
    def __init__(self,midpoint=44,end=88):
        if not 0<midpoint<end:raise ValueError('Invalid phased boost endpoints')
        self.midpoint=midpoint;self.end=end;self.cutoff=None

    def update(self,step,semantic,text):
        if step>=self.midpoint:self.cutoff=self.midpoint
        return dict(semantic_weight=max(0.,1-step/self.midpoint),
            text_weight=min(1.,max(0.,(step-self.midpoint)/(self.end-self.midpoint))),
            phase='semantic' if step<self.midpoint else 'prompt',cutoff=self.cutoff)


def high_attention_factor(values):
    """Only the upper quartile gets a bonus; reaches one at the 95th percentile."""
    q75,q95=torch.quantile(values.float(),torch.tensor([.75,.95],device=values.device))
    factor=((values-q75)/(q95-q75).clamp_min(1e-12)).clamp(0,1)
    return factor,q75,q95


class AttentionPriority:
    def __init__(self,llm,prepared,tokenizer,alpha=.02,mode='dynamic',text_source='revealed',start=40,smoothing=.2,
                 method='legacy_balance',midpoint=44,end=88):
        if not math.isfinite(alpha) or alpha<0 or mode not in {'dynamic','semantic_fixed','text_fixed'} or text_source not in {'revealed','prompt'}:
            raise ValueError('Invalid attention priority configuration')
        if method not in {'legacy_balance','phased_odds'} or (method=='phased_odds' and (text_source!='prompt' or start!=0)):
            raise ValueError('Phased odds requires fixed prompt text and start0')
        self.probe=OnlineAttentionScores(llm,prepared,special_ids=tokenizer.all_special_ids)
        self.prepared=prepared;self.tokenizer=tokenizer;self.alpha=alpha
        self.mode=mode;self.text_source=text_source;self.start=start
        self.method=method
        self.ramp=FixedPhaseRamp(midpoint,end) if method=='phased_odds' else BalanceRamp(start,smoothing)
        self.records=[]

    def __enter__(self):self.probe.__enter__();return self
    def __exit__(self,*args):return self.probe.__exit__(*args)

    def begin(self,x,step,block,block_length):
        self.step=step
        self.probe.begin(x,block,block_length,self.tokenizer.eos_token_id)

    def bonus(self,logits):
        _,mass=self.probe.summaries()
        semantic=mass[:,0];text=mass[:,3 if self.text_source=='revealed' else 1]
        mean=mass.mean(0).cpu().tolist()
        weights=self.ramp.update(self.step,mean[0],mean[3 if self.text_source=='revealed' else 1])
        left,right=weights['semantic_weight'],weights['text_weight']
        if self.mode!='dynamic':
            left=float(self.step>=self.start and self.mode=='semantic_fixed')
            right=float(self.step>=self.start and self.mode=='text_fixed')
        # Within-forward relative source mass, bounded [0,1]. A constant bonus
        # across candidates cannot change their order. This is not calibration.
        candidate=logits[0].index_select(0,self.probe.query_offsets)
        predictions=candidate.argmax(-1)
        confidence=candidate.double().softmax(-1).gather(-1,predictions[:,None])[:,0]
        extra={}
        if self.method=='phased_odds':
            sa,s75,s95=high_attention_factor(semantic)
            ta,t75,t95=high_attention_factor(text)
            shift=self.alpha*(left*sa+right*ta).double()
            # Stable odds multiplication, including exact probabilities0/1.
            adjusted=confidence/(confidence+(1-confidence)*torch.exp(-shift))
            adjusted=torch.where(shift==0,confidence,adjusted)
            values=adjusted-confidence
            extra=dict(log_odds_shift=shift.cpu().tolist(),adjusted_priority=adjusted.cpu().tolist(),
                semantic_attention_factor=sa.cpu().tolist(),prompt_attention_factor=ta.cpu().tolist(),
                attention_thresholds=dict(semantic_q75=float(s75),semantic_q95=float(s95),prompt_q75=float(t75),prompt_q95=float(t95)))
        else:
            sa=semantic/semantic.max().clamp_min(1e-12)
            ta=text/text.max().clamp_min(1e-12)
            values=self.alpha*(left*sa+right*ta)
        bonus=logits.new_zeros(logits.shape[:2],dtype=torch.float64)
        bonus[0,self.probe.query_offsets]=values.double()
        self.records.append(dict(step=self.step,**weights,applied_semantic_weight=left,applied_text_weight=right,
            mean_mass=dict(zip(('semantic','prompt','acoustic','revealed_answer'),mean)),
            revealed_text_keys=len(self.probe.revealed),query_offsets=self.probe.query_offsets.cpu().tolist(),
            per_layer_mean_mass=torch.stack(self.probe.layer_mass).cpu().tolist(),
            per_query_mass=mass.cpu().tolist(),base_top1_confidence=confidence.cpu().tolist(),
            predicted_ids=predictions.cpu().tolist(),priority_bonus=values.cpu().tolist(),**extra))
        return bonus


@torch.no_grad()
def run_priority_study(llm,prepared,baseline,args,tokenizer,row,work):
    from .run import display_text,write_json
    from .inputs import score
    from .benchmark import visible_ids
    alphas=args.priority_strengths
    if not alphas or any(not math.isfinite(a) or not 0<a<=.2 for a in alphas) or len(set(alphas))!=len(alphas):
        raise ValueError('Priority strengths must be unique and in (0,.2]')
    sources=args.priority_text_sources
    method=getattr(args,'priority_method','legacy_balance')
    specs=[dict(alpha=0.,mode='dynamic',text_source=sources[0],method=method)]
    specs += [dict(alpha=a,mode=m,text_source=t,method=method) for t in sources for a in alphas
              for m in ('dynamic','semantic_fixed','text_fixed')]
    report=dict(protocol='attention-commit-priority-v1',policies=[],
        baseline_score=score(display_text(tokenizer,baseline,prepared.prompt.shape[1]),row.get('target'),row.get('metric','none')),
        note='Accuracy experiment; all 32 layers and all currently eligible masked queries. '
             'Priority bonus changes commitment order only, not logits, top1 IDs or number of commits per forward. '
             'phased_odds multiplies odds using upper-quartile source attention and fixed temporal weights; legacy_balance adds probability-space priority. '
             'All runs include attention observation overhead; instrumented times are not clean speed benchmarks. '
             'Mass is summed over source keys. Revealed-text mass can increase mechanically with revealed key count. '
             'No final answer/reference or future baseline length used for the priority rule.')
    expected=visible_ids(baseline,prepared,tokenizer.eos_token_id)
    for spec in specs:
        tag='phased_odds_' if method=='phased_odds' else ''
        name=f"priority_{spec['text_source']}_{tag}{spec['mode']}_a{spec['alpha']:g}"
        print(f'  {name}: starting observed decode',flush=True)
        torch.cuda.synchronize();began=time.perf_counter()
        with AttentionPriority(llm,prepared,tokenizer,**spec,start=args.priority_start,smoothing=args.priority_smoothing,
                               midpoint=args.priority_midpoint,end=args.priority_end) as priority:
            final,_,events=decode(llm,prepared,args.length,args.block_length,args.steps,save_states=False,
                eos_id=tokenizer.eos_token_id,priority=priority,progress_every=args.progress_every,progress_label=name)
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        text=display_text(tokenizer,final,prepared.prompt.shape[1])
        ids=visible_ids(final,prepared,tokenizer.eos_token_id)
        if spec['alpha']==0 and ids!=expected:raise RuntimeError('Zero-priority observed baseline differs from official baseline')
        result=dict(condition=name,**spec,text=text,score=score(text,row.get('target'),row.get('metric','none')),
            forwards=len(events),answer_has_no_eos=tokenizer.eos_token_id not in ids,visible_answer_ids=ids,
            same_visible_tokens=ids==expected,instrumented_seconds=elapsed,cutoff=priority.ramp.cutoff,
            nonzero_bonus_forwards=sum(any(v!=0 for v in r['priority_bonus']) for r in priority.records),
            midpoint=args.priority_midpoint if method=='phased_odds' else None,
            end=args.priority_end if method=='phased_odds' else None)
        for record,event in zip(priority.records,events):record['committed_positions']=event['committed_positions']
        write_json(work/(name+'.json'),dict(result=result,forwards=priority.records))
        report['policies'].append(result)
        write_json(work/'attention_priority.json',report)
        print(f"  {name}: {len(events)} forwards; cutoff={result['cutoff']}; active_bonus_forwards={result['nonzero_bonus_forwards']}; score={result['score']}",flush=True)
        del final
    return dict(file='attention_priority.json',policies=len(specs))
