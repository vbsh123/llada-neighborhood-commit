"""Compare matched completed top1 and top1_window runs without inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .trace_source import Source


def compare_window(baseline, window, out):
    def read_run(run, expected):
        source=Source(Path(run));samples={};config=None;batch_sizes=Counter();neighbors=[];answer_neighbors=[]
        forward_seconds=0
        try:
            for name in source.names:
                result=source.result(name);sid=str(result['sample_id'])
                if sid in samples:raise ValueError('Duplicate question')
                samples[sid]=result
                for r in source.records(name):
                    if r['type']=='header':
                        assert r['config']['policy']==expected
                        comparable={k:v for k,v in r['config'].items() if k!='policy'}
                        if config is not None and comparable!=config:raise ValueError('Mixed configurations')
                        config=comparable
                    if r['type']!='step':continue
                    forward_seconds+=r['forward_seconds'];batch_sizes[len(r['commit_positions'])]+=1
                    if expected=='top1_window':
                        by_position={row['position']:row for row in r['positions']}
                        anchor=r['selection']['anchor_position']
                        for p in r['commit_positions']:
                            if p!=anchor:
                                confidence=by_position[p]['confidence'];neighbors.append(confidence)
                                if p<result['answer_token_length'] and not by_position[p]['special']:
                                    answer_neighbors.append(confidence)
        finally:source.close()
        n=len(samples)
        if not n:raise ValueError('No completed samples')
        summary={'samples':n,'numeric_accuracy':sum(s['correct_numeric'] for s in samples.values())/n,
                 'total_forwards':sum(s['steps'] for s in samples.values()),
                 'mean_forwards_per_question':sum(s['steps'] for s in samples.values())/n,
                 'total_elapsed_seconds':sum(s['elapsed_seconds'] for s in samples.values()),
                 'total_forward_seconds':forward_seconds,'commit_batch_histogram':dict(sorted(batch_sizes.items())),
                 'mean_commits_per_forward':sum(k*v for k,v in batch_sizes.items())/sum(batch_sizes.values()),
                 'length_limit_count':sum(s['hit_length_limit'] for s in samples.values()),
                 'forced_neighbor_count':len(neighbors),
                 'forced_neighbors_below_0_9':sum(p<.9 for p in neighbors),
                 'mean_forced_neighbor_confidence':sum(neighbors)/len(neighbors) if neighbors else None}
        summary['forced_answer_neighbors']={'count':len(answer_neighbors),
            'below_0_9':sum(p<.9 for p in answer_neighbors),
            'mean_confidence':sum(answer_neighbors)/len(answer_neighbors) if answer_neighbors else None,
            'definition':'Non-anchor commits, nonspecial and before the final first stop.'}
        return samples,config,summary
    a,ac,sa=read_run(baseline,'top1');b,bc,sb=read_run(window,'top1_window')
    if set(a)!=set(b) or ac!=bc:raise ValueError('Runs must have identical samples and settings except policy')
    changes=Counter()
    for sid in a:
        if (a[sid]['question'],a[sid]['reference'])!=(b[sid]['question'],b[sid]['reference']):
            raise ValueError('Question/reference mismatch')
        changes['both_correct' if a[sid]['correct_numeric'] and b[sid]['correct_numeric'] else
                'window_only_correct' if b[sid]['correct_numeric'] else
                'baseline_only_correct' if a[sid]['correct_numeric'] else 'both_wrong']+=1
    report={'top1':sa,'top1_window':sb,'paired_correctness':dict(changes),
            'forward_reduction_percent':100*(1-sb['total_forwards']/sa['total_forwards']),
            'numeric_accuracy_change_percentage_points':100*(sb['numeric_accuracy']-sa['numeric_accuracy']),
            'limits':'Full response window filled, including special/post-stop slots. Wall time includes trace collection overhead. Neighbor confidence does not establish correctness.'}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True);p.add_argument('--window',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();compare_window(a.baseline,a.window,a.out)
