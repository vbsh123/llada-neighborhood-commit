"""Inspect four-token successes that become three-token failures; no inference."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path


def inspect_window_failures(four, three, out):
    """Read result.json only; require identical question sets and references."""
    def load(root):
        results={}
        for path in sorted(Path(root).glob('samples/*/result.json')):
            row=json.loads(path.read_text());sid=str(row['sample_id'])
            if sid in results:raise ValueError('Duplicate sample ID')
            results[sid]=row
        if not results:raise ValueError(f'No completed results in {root}')
        return results
    a=load(four);b=load(three)
    if set(a)!=set(b):raise ValueError('Completed sample sets differ')
    counts=Counter();cases=[];reverse=[]
    for sid in sorted(a):
        x,y=a[sid],b[sid]
        if (x['question'],x['reference'])!=(y['question'],y['reference']):
            raise ValueError(f'Question/reference mismatch for {sid}')
        key=('both_correct' if x['correct_numeric'] and y['correct_numeric'] else
             'four_correct_three_wrong' if x['correct_numeric'] else
             'three_correct_four_wrong' if y['correct_numeric'] else 'both_wrong')
        counts[key]+=1
        if key in ('four_correct_three_wrong','three_correct_four_wrong'):
            def details(r):
                return {k:r.get(k) for k in ('answer','predicted_answer','gold_answer',
                        'answer_extraction','correct_numeric','steps','answer_token_length')}
            case={'sample_id':sid,'question':x['question'],'reference':x['reference'],
                  'four_tokens':details(x),'three_tokens':details(y)}
            (cases if key=='four_correct_three_wrong' else reverse).append(case)
    report={'samples':len(a),'counts':dict(counts),'four_correct_three_wrong':cases,
            'three_correct_four_wrong':reverse,
            'definition':'Correctness uses the saved numeric-answer scorer; read texts to check extraction errors. No traces read or model inference.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'cases.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>Four-token wins, three-token failures</title>',
          '<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:15px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;background:#f6f7f9}article{border-top:1px solid #ccc;margin:35px 0} @media(max-width:750px){.pair{grid-template-columns:1fr}}</style>',
          f'<h1>Four-token correct → three-token wrong</h1><p>{len(cases)} cases out of {len(a)} matching questions. Reverse flips: {len(reverse)}.</p>',
          '<p>These labels use numeric answer extraction; inspect the texts for scoring mistakes. This comparison does not establish which commitment caused a failure.</p>']
    for case in cases:
        page.append(f'<article><h2>Sample {html.escape(case["sample_id"])}</h2><p>{html.escape(case["question"])}</p><div class="pair">')
        for key,label in [('four_tokens','Four tokens — scored correct'),('three_tokens','Three tokens — scored wrong')]:
            row=case[key]
            page.append(f'<section><h3>{label}</h3><p>Extracted: {html.escape(str(row["predicted_answer"]))}; gold: {html.escape(str(row["gold_answer"]))}; method: {html.escape(str(row["answer_extraction"]))}</p><pre>{html.escape(row["answer"] or "")}</pre></section>')
        page.append(f'</div><details><summary>Reference solution</summary><pre>{html.escape(case["reference"])}</pre></details></article>')
    (out/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print(json.dumps({'samples':len(a),'counts':dict(counts),'out':str(out)},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--four',required=True);p.add_argument('--three',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();inspect_window_failures(a.four,a.three,a.out)
