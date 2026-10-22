"""Paired, forward-only removal sensitivity at retrospective content checkpoints."""
import json
import math

import torch

from .attention_scores import OnlineAttentionScores
from .decode import forward_logits
from .hooks import AttentionProbe,blocks_of


def checkpoint_states(states,prepared,answer_ids,special_ids,eos_id,fractions):
    stop=answer_ids.index(eos_id) if eos_id in answer_ids else len(answer_ids)
    content=[i for i,t in enumerate(answer_ids[:stop]) if t not in set(special_ids)]
    if not content: return [],'No baseline answer-content positions'
    result=[]
    for fraction in fractions:
        found=None
        for step,x in enumerate(states):
            answer=x[0,prepared.prompt.shape[1]:]
            revealed=sum(int(answer[i])!=prepared.mask_id for i in content)/len(content)
            if fraction<=revealed<1 and bool((answer==prepared.mask_id).any()):
                found=dict(step=step,requested_content_fraction=fraction,actual_content_fraction=revealed)
                break
        if found is not None:result.append(found)
    return result,None


def prediction_change(reference,changed):
    """CPU query-only logits; float32 log probabilities, small vocab chunks."""
    kl=[];flips=[];loss=[];entropy=[];base_probability=[]
    for a,b in zip(reference.split(8),changed.split(8)):
        a=a.float().log_softmax(-1);b=b.float().log_softmax(-1)
        index=a.argmax(-1,keepdim=True)
        p=a.exp()
        kl.extend((p*(a-b)).sum(-1).clamp_min(0).tolist())
        flips.extend((index[:,0]!=b.argmax(-1)).tolist())
        pa=p.gather(-1,index)[:,0];pb=b.exp().gather(-1,index)[:,0]
        loss.extend((pa-pb).tolist());base_probability.extend(pa.tolist())
        entropy.extend((-(p*a).sum(-1)).tolist())
    n=len(kl)
    if not n:return dict(query_count=0,kl_mean=None,top1_change_rate=None,
        baseline_top1_probability_mean=None,baseline_top1_probability_drop_mean=None,baseline_entropy_mean=None,
        per_query_kl=[],per_query_top1_changed=[],per_query_probability_drop=[])
    return dict(query_count=n,kl_mean=sum(kl)/n,top1_change_rate=sum(flips)/n,
                baseline_top1_probability_mean=sum(base_probability)/n,
                baseline_top1_probability_drop_mean=sum(loss)/n,baseline_entropy_mean=sum(entropy)/n,
                per_query_kl=kl,per_query_top1_changed=flips,per_query_probability_drop=loss)


@torch.no_grad()
def collect_sensitivity(llm,prepared,states,events,answer_ids,tokenizer,args,work):
    from .run import write_json
    fractions=args.sensitivity_fractions;budgets=args.sensitivity_budgets
    if (not fractions or fractions!=sorted(set(fractions)) or any(not math.isfinite(f) or not 0<f<1 for f in fractions)
            or not budgets or min(budgets)<1 or not 0<=args.sensitivity_layer<len(blocks_of(llm))):
        raise ValueError('Invalid removal sensitivity settings')
    points,reason=checkpoint_states(states,prepared,answer_ids,tokenizer.all_special_ids,tokenizer.eos_token_id,fractions)
    stop=answer_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in answer_ids else len(answer_ids)
    content={i for i,t in enumerate(answer_ids[:stop]) if t not in set(tokenizer.all_special_ids)}
    layers=list(range(args.sensitivity_layer,len(blocks_of(llm))))
    write_json(work/'removal_sensitivity_config.json',dict(checkpoints=points,skip_reason=reason,budgets=budgets,
        layers=layers,query_population='All currently masked positions eligible in current decoder block; identical within checkpoint',
        note='Retrospective baseline-content checkpoints, not an online policy. Selection uses all online-eligible masked queries. '
             'Primary metrics use their baseline lexical-content subset, excluding padding/EOS; all-eligible metrics also saved. '
             'No checkpoints after content completion. Separate one-forward KV-column removal per condition; no speed claim. '
             'Fixed-early and re-ranked selections. No hidden-state storage.'))
    fixed={};total=0
    with (work/'removal_sensitivity.jsonl').open('w') as output:
        for point in points:
            step=point['step'];x=states[step].to(prepared.prompt.device)
            block=events[step]['block']
            with OnlineAttentionScores(llm,prepared,layers,tokenizer.all_special_ids) as probe:
                probe.begin(x,block,args.block_length,tokenizer.eos_token_id)
                logits=forward_logits(llm,prepared,x)
                offsets=probe.query_offsets
                content_local=torch.tensor([i for i,p in enumerate(offsets.cpu().tolist()) if p in content],dtype=torch.long)
                reference=logits[0].index_select(0,offsets).cpu();del logits
                key_scores,_=probe.summaries();key_scores=key_scores.cpu()
            print(f"  sensitivity checkpoint F{step}: content={point['actual_content_fraction']:.1%}, queries={len(offsets)}",flush=True)
            measured={}
            for group in ('acoustic','semantic','prompt','prefix'):
                pool=prepared.group_positions(group)
                ranked=sorted(pool,key=lambda p:(float(key_scores[p]),p))
                for count in budgets:
                    if count>len(pool):
                        record=dict(**point,group=group,budget=count,skipped='Budget exceeds group size')
                        output.write(json.dumps(record)+'\n');output.flush();continue
                    fixed.setdefault((group,count),ranked[:count])
                    for mode in ('reranked','fixed_early'):
                        positions=ranked[:count] if mode=='reranked' else fixed[group,count]
                        identity=tuple(sorted(positions))
                        reused=identity in measured
                        if not reused:
                            with AttentionProbe(llm,positions,layers=layers,mode='remove'):
                                logits=forward_logits(llm,prepared,x)
                                changed=logits[0].index_select(0,offsets).cpu();del logits
                            measured[identity]=dict(
                                content=prediction_change(reference.index_select(0,content_local),changed.index_select(0,content_local)),
                                all_eligible=prediction_change(reference,changed))
                        record=dict(**point,group=group,budget=count,selection=mode,removed_positions=positions,
                            reused_identical_intervention=reused,
                            query_offsets=offsets.cpu().tolist(),removed_attention_mass=float(key_scores[positions].sum()),
                            content_query_offsets=[int(offsets[i]) for i in content_local.tolist()],
                            removed_by_group={g:len(set(positions)&set(prepared.group_positions(g))) for g in ('acoustic','semantic','prompt')},
                            metrics=measured[identity]['content'],all_eligible_metrics=measured[identity]['all_eligible'])
                        output.write(json.dumps(record,allow_nan=False)+'\n');output.flush()
                        total+=1
                print(f'    {group}: finished paired removals',flush=True)
            del reference,x
    return dict(measurements=total,checkpoints=points,file='removal_sensitivity.jsonl',skip_reason=reason)
