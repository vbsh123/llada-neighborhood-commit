"""Use vendored upstream implementations, not a new extraction/normalization rule."""
import hashlib
import json
from pathlib import Path

from .vendor.snapshot import RegexFilter, TakeFirstFilter, exact_match_hf_evaluate

HERE = Path(__file__).resolve().parent/'vendor'
COMMIT = '84aa9f95fea2e1bd298e1859cab0b12094f80e0b'


def load_protocol():
    provenance=json.loads((HERE/'provenance.json').read_text())
    if provenance['commit']!=COMMIT:
        raise ValueError('Unexpected harness revision')
    for name,expected in provenance['bundled_file_sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Pinned evaluator file changed: {name}')
    task=json.loads((HERE/'task.json').read_text())
    metric=dict(task['metric_list'][0])
    if metric.pop('metric')!='exact_match':
        raise ValueError('Unexpected harness metric')
    metric.pop('aggregation'); metric.pop('higher_is_better')
    filters={}
    for pipeline in task['filter_list']:
        regex,first=pipeline['filter']
        regex=dict(regex)
        if regex.pop('function')!='regex' or first!={'function':'take_first'}:
            raise ValueError('Unexpected harness filter pipeline')
        filters[pipeline['name']]=RegexFilter(**regex)
    return provenance,metric,filters


def score_saved(answers,references):
    if len(answers)!=len(references) or not answers:
        raise ValueError('Require nonempty, equally sized answers/references')
    if any(not isinstance(s,str) for s in [*answers,*references]):
        raise ValueError('Answers/references must be strings')
    _,metric,filters=load_protocol()
    docs=[{'answer':r} for r in references]
    golds=list(TakeFirstFilter().apply(filters['strict-match'].apply([[r] for r in references],docs),docs))
    if any(g==filters['strict-match'].fallback for g in golds):
        raise ValueError('Reference lacks the pinned GSM8K #### numeric answer format')
    output={}
    for name,extractor in filters.items():
        predictions=list(TakeFirstFilter().apply(extractor.apply([[a] for a in answers],docs),docs))
        correct=[bool(exact_match_hf_evaluate(predictions=[p],references=[r],**metric)['exact_match'])
                 for p,r in zip(predictions,references)]
        accuracy=float(exact_match_hf_evaluate(predictions=predictions,references=references,**metric)['exact_match'])
        output[name]={'predictions':predictions,'correct':correct,'accuracy':accuracy,
                      'invalid_count':sum(p==extractor.fallback for p in predictions)}
    return output
