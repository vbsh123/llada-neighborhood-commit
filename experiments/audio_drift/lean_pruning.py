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
    specs=policy_specs(args,'process-warmup')
    if getattr(args,'pruning_sweep',False):
        # Warm the shared implementation once, not all ten configurations.
        source=next(s for s in specs if s['pruning']['control']=='audio')
        specs=[source,next(s for s in specs if s.get('control_source')==source['condition'])]
    if getattr(args,'cache_ramp',False):
        specs=[next(s for s in specs if s['condition']=='ramp_control_a16_p8_s8')]
    print(f'One-time warmup: baseline and {len(specs)} policies (untimed)',flush=True)
    final,_,_=decode(llm,prepared,args.length,args.block_length,args.steps,
                     save_states=False,eos_id=tokenizer.eos_token_id)
    del final
    completed=[]
    for spec in specs:
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
             'scoring, output writes and accounting serialization. Baseline first; each audio source precedes its generic control. '
             'When present, generic uses audio event times/budgets and ranks its own prefix. '
             'Cache-only conditions use deterministic shuffled timing order. No claim of identical-input timing.')
    for spec in policy_specs(args,row['id']):
        spec=resolve_control(spec,report['policies'])
        print(f"  single-pass {spec['condition']}: starting",flush=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter()
        with policy_context(llm,prepared,spec) as engine:
            final,_,events=decode(llm,prepared,args.length,args.block_length,args.steps,
                save_states=False,eos_id=tokenizer.eos_token_id,
                progress_every=args.progress_every,progress_label='policy')
        torch.cuda.synchronize();seconds=time.perf_counter()-began
        peak=torch.cuda.max_memory_allocated()
        accounting=engine.accounting()
        if spec.get('pruning') and (getattr(args,'pruning_sweep',False) or getattr(args,'minimal_pruning_records',False)):
            # The matched control needs check times and removal budgets, not
            # full per-position attention histories. Preserve only decisions.
            account=accounting['pruning']
            account['trigger_checks']=[{k:r[k] for k in ('step','apply_step','current_key_count',
                'eligible_by_group','planned_removals')} for r in account['trigger_checks']]
            for pick in account['selections']:
                pick.pop('probe_positions',None)
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
        if not (getattr(args,'pruning_sweep',False) or getattr(args,'minimal_pruning_records',False)):
            write_json(work/(spec['condition']+'_steps.json'),events)
        write_json(work/'cache_benchmark.json',report)
        print(f"    {seconds:.2f}s; {accounting['forwards']} forwards; "
              f"{result['visible_answer_tokens']/seconds:.2f} TPS; score={result['score']}",flush=True)
        del final
    return report
