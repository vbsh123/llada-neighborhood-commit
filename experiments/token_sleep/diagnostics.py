"""Local layer parity with identical inputs/cache, in native dtype and FP32.

This is a diagnostic, never an automatic tolerance override or timing benchmark.
Only one block's weights are promoted at a time and restored in a finally clause.
"""
from .selective import SelectiveForward
from .simulation import FrozenStates, SleepSchedule, checked_blocks


def difference(a,b):
    a,b=a.float(),b.float()
    return {'max_absolute_difference':(a-b).abs().max().item(),
            'mean_absolute_difference':(a-b).abs().mean().item()}


def compare_layer(model,block,inputs,seed,sleepers):
    """Both computations receive identical current layer inputs and old states.

    Optimized errors from previous layers are not propagated into this comparison.
    This separates local operator differences from cross-layer error accumulation.
    """
    import torch
    dtype=next(block.parameters()).dtype
    n=inputs.shape[1]
    active=torch.tensor([p for p in range(n) if p not in set(sleepers)],device=inputs.device)
    old={key:t.to(dtype=dtype).clone() for key,t in seed.items()}
    inputs=inputs.to(dtype=dtype)
    backend=SelectiveForward(model,blocks=[block])
    backend.kv[0]=(old['k'].clone(),old['v'].clone())
    stages={}; phase=['reference']; handles=[]
    for name in ('attn_norm','q_proj','k_proj','v_proj','attn_out','ff_norm','ff_proj','up_proj','ff_out'):
        module=getattr(block,name)
        def capture(_module,_args,output,_name=name):
            stages[(phase[0],_name)]=output.detach().clone()
        handles.append(module.register_forward_hook(capture))
    try:
        with FrozenStates([block]) as frozen:
            frozen.cache[0]={key:t.clone() for key,t in old.items()}
            frozen.begin(sleepers)
            reference,_=block(inputs,use_cache=False)
            frozen.end()
        phase[0]='selective'
        with backend.attention_methods(active,n):
            actual,_=block(inputs.index_select(1,active),use_cache=False)
        return {'output':difference(actual,reference.index_select(1,active)),
                'stages':{name:difference(stages[('selective',name)],
                    stages[('reference',name)].index_select(1,active))
                    for name in ('attn_norm','q_proj','k_proj','v_proj','attn_out','ff_norm','ff_proj','up_proj','ff_out')}}
    finally:
        for handle in handles: handle.remove()


def diagnose(model,tokenizer,sample,config):
    import torch
    from confidence_geography.core import select
    from .benchmark import checked_input,compact_rows
    blocks=checked_blocks(model)
    x,pl=checked_input(model,tokenizer,sample['question'],config['length'],config['mask_id'])
    if config['length']<3: raise ValueError('Layer diagnosis needs response length >=3')
    schedule=SleepSchedule(config['length'],'simulated_sleep',confidence=config['sleep_confidence'],
        high_entropy=config['high_entropy'],high_sleep=config['high_sleep'],low_sleep=0)
    current_inputs={}; handles=[]
    with torch.inference_mode(),FrozenStates(blocks) as frozen:
        frozen.begin([]); initial=model(x,use_cache=False).logits; frozen.end()
        seed=[{key:t.clone() for key,t in entry.items()} for entry in frozen.cache]
        rows=compact_rows(initial[:,pl:],list(range(config['length'])),config['mask_id'])
        chosen=select(rows,config['policy'],1,config['commit_threshold'])
        committed={r['position'] for r in chosen}
        schedule.prepare(0,x[0,pl:].tolist(),config['mask_id'])
        schedule.finish(0,rows,committed)
        for row in chosen: x[0,pl+row['position']]=row['token_id']
        sleepers=schedule.prepare(1,x[0,pl:].tolist(),config['mask_id'])
        source='actual_first_sleep_rule'
        if not sleepers:
            sleepers=[p for p in range(0,config['length'],3) if p not in committed]
            source='synthetic_sparse_pattern_no_actual_sleepers_at_first_step'
        absolute=[pl+p for p in sleepers]
        del initial
        try:
            for index,block in enumerate(blocks):
                def capture_input(_module,args,_index=index):
                    current_inputs[_index]=args[0].detach().clone()
                handles.append(block.register_forward_pre_hook(capture_input))
            frozen.begin(absolute); model(x,use_cache=False); frozen.end()
        finally:
            for handle in handles: handle.remove()
    report={'scope':'each layer receives identical simulator inputs and identical seeded K/V',
        'sleep_source':source,'sleep_positions':sleepers,'layers':[],
        'fp32_local_output_tolerance':.0001,
        'note':'No generations or timing benchmark. One block at a time promoted; native weights restored. Local parity does not establish global quality or speed.'}
    with torch.inference_mode():
        for index,block in enumerate(blocks):
            dtypes={p.dtype for p in block.parameters()}
            if len(dtypes)!=1: raise ValueError('Mixed parameter dtypes in diagnostic block are unsupported')
            original_dtype=next(iter(dtypes))
            native=compare_layer(model,block,current_inputs[index],seed[index],absolute)
            try:
                block.to(dtype=torch.float32)
                fp32=compare_layer(model,block,current_inputs[index],seed[index],absolute)
            finally:
                block.to(dtype=original_dtype)
            entry={'layer':index,'native_dtype':str(original_dtype),'native':native,'fp32':fp32}
            report['layers'].append(entry)
            print(f"layer={index} native_max={native['output']['max_absolute_difference']:.8g} fp32_max={fp32['output']['max_absolute_difference']:.8g}",flush=True)
    report['max_native_local_output_difference']=max(r['native']['output']['max_absolute_difference'] for r in report['layers'])
    report['max_fp32_local_output_difference']=max(r['fp32']['output']['max_absolute_difference'] for r in report['layers'])
    report['all_fp32_local_layers_within_tolerance']=report['max_fp32_local_output_difference']<=report['fp32_local_output_tolerance']
    return report
