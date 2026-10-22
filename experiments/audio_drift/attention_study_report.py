"""Reports for forward sensitivity and attention-based commitment priorities."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def analyze(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    sensitivity=[];priority=[];trajectories=[]
    for path in sorted((root/'samples').glob('*/result.json')):
        if path.parent.name.startswith('.'):continue
        result=json.loads(path.read_text());folder=path.parent
        measurement=folder/'removal_sensitivity.jsonl'
        if measurement.exists():
            for line in measurement.read_text().splitlines():
                row=json.loads(line)
                if row.get('skipped'):continue
                sensitivity.append(dict(id=result['id'],cohort=result.get('cohort','unspecified'),baseline_correct=result['score'].get('correct'),
                    requested_content_fraction=row['requested_content_fraction'],actual_content_fraction=row['actual_content_fraction'],
                    step=row['step'],group=row['group'],budget=row['budget'],selection=row['selection'],
                    removed_attention_mass=row['removed_attention_mass'],
                    **{k:v for k,v in row['metrics'].items() if not isinstance(v,list)}))
        summary=folder/'attention_priority.json'
        if summary.exists():
            for p in json.loads(summary.read_text())['policies']:
                priority.append(dict(id=result['id'],cohort=result.get('cohort','unspecified'),condition=p['condition'],baseline_correct=result['score'].get('correct'),
                    policy_correct=p['score'].get('correct'),parse_failed=p['score'].get('parse_failed'),
                    forwards=p['forwards'],cutoff=p['cutoff'],no_eos=p['answer_has_no_eos'],
                    instrumented_seconds=p['instrumented_seconds']))
                raw=json.loads((folder/(p['condition']+'.json')).read_text())
                for r in raw['forwards']:
                    trajectories.append(dict(id=result['id'],condition=p['condition'],step=r['step'],phase=r['phase'],
                        semantic_mass=r['mean_mass']['semantic'],prompt_mass=r['mean_mass']['prompt'],
                        revealed_answer_mass=r['mean_mass']['revealed_answer'],revealed_text_keys=r['revealed_text_keys'],
                        semantic_weight=r['applied_semantic_weight'],text_weight=r['applied_text_weight'],
                        semantic_ema=r.get('semantic_ema'),text_ema=r.get('text_ema'),
                        semantic_change_per_forward=r.get('semantic_change_per_forward'),
                        text_change_per_forward=r.get('text_change_per_forward'),
                        margin_change_per_forward=r.get('margin_change_per_forward')))
    write_csv(out/'removal_sensitivity.csv',sensitivity)
    write_csv(out/'attention_priority_pairs.csv',priority)
    write_csv(out/'attention_priority_trajectories.csv',trajectories)
    lines=['# Attention intervention study','',
        'Development results. Forward removal sensitivity is not final-answer accuracy or speed. '
        'Query identities/difficulty can change across checkpoints; baseline entropy and query counts are retained. '
        'Fixed-early selection controls token identity, not changing query population. '
        'Priority results change commitment order only; instrumented times include attention scoring and are not clean speed benchmarks.','']
    if sensitivity:
        lines += [f"Forward sensitivity: {len({r['id'] for r in sensitivity})} examples, {len(sensitivity)} measurements. See removal_sensitivity.csv and raw per-query JSONL.",'']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        budgets=sorted({r['budget'] for r in sensitivity})
        fig,axes=plt.subplots(2,len(budgets),squeeze=False,figsize=(5*len(budgets),8))
        for i,selection in enumerate(('reranked','fixed_early')):
            for j,budget in enumerate(budgets):
                ax=axes[i,j]
                for group in ('acoustic','semantic','prompt','prefix'):
                    grouped=defaultdict(list)
                    for r in sensitivity:
                        if r['selection']==selection and r['budget']==budget and r['group']==group and r['query_count']:
                            grouped[r['requested_content_fraction']].append(r)
                    xs=sorted(grouped)
                    if xs:ax.plot([100*x for x in xs],[statistics.mean(r['kl_mean'] for r in grouped[x]) for x in xs],'-o',label=group)
                ax.set_title(f'{selection}: remove {budget} positions');ax.set_xlabel('Requested baseline content checkpoint (%)')
                ax.set_ylabel('Mean KL per eligible masked query');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Forward-only removal sensitivity; examples equally weighted, layers fixed within run')
        fig.tight_layout();fig.savefig(out/'removal_sensitivity.png',dpi=160);plt.close(fig)
    if priority:
        lines += ['| Policy | N | Baseline correct | Policy correct | Correct → wrong | Wrong → correct | Cutoff observed | Parse failures |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for condition in sorted({r['condition'] for r in priority}):
            v=[r for r in priority if r['condition']==condition]
            lines.append(f"| {condition} | {len(v)} | {sum(bool(r['baseline_correct']) for r in v)} | "
                         f"{sum(bool(r['policy_correct']) for r in v)} | {sum(bool(r['baseline_correct']) and not r['policy_correct'] for r in v)} | "
                         f"{sum(not r['baseline_correct'] and bool(r['policy_correct']) for r in v)} | {sum(r['cutoff'] is not None for r in v)} | "
                         f"{sum(bool(r['parse_failed']) for r in v)} |")
    (out/'attention_study_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    print(f'Attention study report: {len(sensitivity)} sensitivity rows, {len(priority)} priority results',flush=True)
    return sensitivity,priority


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run)
