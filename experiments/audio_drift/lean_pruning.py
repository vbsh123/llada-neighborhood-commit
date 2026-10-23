"""One timed generation per policy; no per-example warmups or replay."""
import time

import torch

from .benchmark import policy_specs, visible_ids
from .compact import policy_context
from .crossing_pruning import resolve_control
from .decode import decode
from .inputs import score


@torch.no_grad()
def warmup(llm,prepared,args,tokenizer):
    print('One-time warmup: baseline, audio pruning, generic control (untimed)',flush=True)
    final,_,_=decode(llm,prepared,args.length,args.block_length,args.steps,
                     save_states=False,eos_id=tokenizer.eos_token_id)
    del final
    completed=[]
    for spec in policy_specs(args,'process-warmup'):
        spec=resolve_control(spec,completed)
        with policy_context(llm,prepared,spec) as engine:
            final,_,_=decode(llm,prepared,args.length,args.block_length,args.steps,
                             save_states=False,eos_id=tokenizer.eos_token_id)
        completed.append(dict(condition=spec['condition'],accounting=[engine.accounting()]))
        del final
    torch.cuda.synchronize()


@torch.no_grad()
def evaluate(llm,prepared,baseline,baseline_seconds,baseline_events,args,tokenizer,row,work,
             preparation_seconds):
    from .run import display_text,write_json
    expected=visible_ids(baseline,prepared,tokenizer.eos_token_id)
    report=dict(protocol='single-pass-eos-v1',stopping='eos',timed_repeats_per_policy=1,
        baseline_seconds=[baseline_seconds],baseline_median_seconds=baseline_seconds,
        baseline_forwards=len(baseline_events),baseline_visible_answer_tokens=len(expected)-int(tokenizer.eos_token_id in expected),
        shared_audio_preparation_seconds=preparation_seconds,policies=[],
        note='One synchronized decode per condition, after one process-wide warmup. '
             'median_seconds is a compatibility field equal to the single measured duration. '
             'No repeated baseline, per-example warmup, replay, or full-canvas state saving. '
             'Includes pruning observations/selection and token commitment. Excludes audio preparation, '
             'scoring, output writes and accounting serialization. Baseline -> audio -> generic order; '
             'generic uses audio event times/budgets and ranks its own prefix. No claim of identical-input timing.')
    for spec in policy_specs(args,row['id']):
        spec=resolve_control(spec,report['policies'])
        print(f"  single-pass {spec['condition']}: starting",flush=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter()
        with policy_context(llm,prepared,spec) as engine:
            final,_,events=decode(llm,prepared,args.length,args.block_length,args.steps,
                save_states=False,eos_id=tokenizer.eos_token_id,
                progress_every=args.progress_every,progress_label='pruning')
        torch.cuda.synchronize();seconds=time.perf_counter()-began
        peak=torch.cuda.max_memory_allocated()
        accounting=engine.accounting()
        for pick in accounting['pruning']['selections']:
            anchor=events[pick['step']]
            pick['anchor_progress']={k:anchor[k] for k in ('remaining_masks','remaining_fraction',
                'active_remaining_masks','active_remaining_fraction','block','step_in_block','previous_changed_tokens')}
            pick['anchor_progress']['revealed_fraction']=1-anchor['remaining_fraction']
        text=display_text(tokenizer,final,prepared.prompt.shape[1])
        ids=visible_ids(final,prepared,tokenizer.eos_token_id)
        result=dict(**spec,seconds=[seconds],median_seconds=seconds,accounting=[accounting],
            peak_allocated_bytes=peak,text=text,visible_answer_ids=ids,same_visible_tokens=ids==expected,
            score=score(text,row.get('target'),row.get('metric','none')),
            answer_has_no_eos=tokenizer.eos_token_id not in ids,
            visible_answer_tokens=len(ids)-int(tokenizer.eos_token_id in ids),
            selected_positions={s['group']:prepared.group_positions(s['group']) for s in spec['schedule']},
            decode_speedup=baseline_seconds/seconds,
            same_forward_count_as_baseline=accounting['forwards']==len(baseline_events))
        report['policies'].append(result)
        write_json(work/(spec['condition']+'_steps.json'),events)
        write_json(work/'cache_benchmark.json',report)
        print(f"    {seconds:.2f}s; {accounting['forwards']} forwards; "
              f"{result['visible_answer_tokens']/seconds:.2f} TPS; score={result['score']}",flush=True)
        del final
    return report
