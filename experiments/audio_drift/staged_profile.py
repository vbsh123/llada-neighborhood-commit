"""Separate operator attribution for compact staged pruning; not latency benchmarks."""
from contextlib import nullcontext
import torch

from .benchmark import policy_specs
from .compact import policy_context
from .decode import forward_logits
from .profile import decoder_scopes


@torch.no_grad()
def profile_staged(llm,prepared,states,active_forwards,args,work,sample_id):
    from .run import write_json
    specs=[s for s in policy_specs(args,sample_id) if s.get('pruning',{}).get('engine')=='compact_staged']
    if not specs:raise ValueError('Pruning operator profile needs staged policies')
    minimum=min(s['pruning']['fraction'] for s in specs)
    # Attribute all six paired patterns at the smaller fraction, rather than
    # repeat the expensive trace capture for every budget.
    specs=[None,*[s for s in specs if s['pruning']['fraction']==minimum]]
    count=min(active_forwards,len(states),max(s['start'] for p in specs if p for s in p['pruning']['stages'])+2)
    inputs=[x.to(prepared.prompt.device) for x in states[:count]]
    directory=work/'pruning_traces';directory.mkdir(exist_ok=True)
    index=[]
    for spec in specs:
        name=spec['condition'] if spec else 'uncached'
        starts=[s['start'] for s in spec['pruning']['stages']] if spec else [args.acoustic_start,args.later_start]
        chosen={0,*starts,*[s+1 for s in starts]}
        with (policy_context(llm,prepared,spec) if spec else nullcontext()) as engine:
            for step,x in enumerate(inputs):
                if step not in chosen:
                    y=forward_logits(llm,prepared,x);del y
                    continue
                torch.cuda.synchronize()
                if engine:engine.profile_scopes=True
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True) as profiler:
                    with decoder_scopes(llm):
                        y=forward_logits(llm,prepared,x);del y
                    torch.cuda.synchronize()
                if engine:engine.profile_scopes=False
                filename=f'{name}_step{step:03d}'
                profiler.export_chrome_trace(str(directory/(filename+'.json')))
                averages=profiler.key_averages()
                (directory/(filename+'.txt')).write_text(averages.table(sort_by='self_cuda_time_total',row_limit=50))
                # CPU annotation aggregates contain attributed device work.
                # Do not duplicate same-name CUDA annotation spans.
                scopes=[dict(name=e.key,calls=e.count,cpu_total_us=e.cpu_time_total,device_total_us=e.device_time_total)
                        for e in averages if e.key.startswith(('decoder.','prune.')) and str(e.device_type).endswith('.CPU')]
                index.append(dict(condition=name,step=step,trace=filename+'.json',scopes=scopes,
                                  execution=engine.events[-1] if engine else None))
                write_json(directory/'index.json',dict(captures=index,note='Instrumented attribution, not clean latency. '
                    'Selection, compaction and output restoration scopes are disjoint from projection scopes. '
                    'Norms, activation, other kernels and dispatch also contribute; do not add scopes to their child kernels.'))
    print(f'  pruning operator traces: {len(index)} captures',flush=True)
