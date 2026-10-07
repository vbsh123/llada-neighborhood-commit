"""One final repair forward over low-confidence forced neighbors in saved runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .core import numeric_answer
from .run import dump, load_model, run_lock, score_logits, source_fingerprint, token_info
from .trace_source import Source


def repair_candidates(records, result, threshold):
    """Forced neighbors only, original confidence < threshold, before final stop.

    Exclude originally special tokens and anchors. No cap, no other confidence
    filter. Selection uses saved data and never consults correctness or gold.
    """
    if not 0<=threshold<=1:raise ValueError('Threshold must be in [0,1]')
    header=None;selected={};seen=set()
    for record in records:
        if record['type']=='header':
            header=record
            if record['config']['policy']!='top1_window':raise ValueError('Window traces required')
            continue
        if record['type']!='step':continue
        commits=record['commit_positions'];anchor=record['selection']['anchor_position']
        if anchor not in commits:raise ValueError('Missing anchor commitment')
        rows={row['position']:row for row in record['positions']}
        for p in commits:
            if p in seen:raise ValueError('Position committed more than once')
            seen.add(p)
            row=rows[p]
            if p!=anchor and row['confidence']<threshold and p<result['answer_token_length'] and not row['special']:
                if result['final_ids'][p]!=row['token_id']:raise ValueError('Final token differs from committed token')
                selected[p]={'position':p,'original_token_id':row['token_id'],
                             'original_confidence':row['confidence'],'commit_step':record['step'],
                             'offset_from_anchor':p-anchor}
    if header is None:raise ValueError('Missing trace header')
    return header,[selected[p] for p in sorted(selected)]


def repaired_result(original, tokens, tokenizer, rows, repair_seconds, forward_seconds):
    """Score using the fixed ORIGINAL stop boundary; allow all non-MASK predictions."""
    result=dict(original);end=original['answer_token_length']
    answer=tokenizer.decode(tokens[:end],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    predicted,method=numeric_answer(answer);gold,_=numeric_answer(original['reference'],reference=True)
    specials=set(tokenizer.all_special_ids);dictionary=dict(original['token_dictionary'])
    for row in rows:
        dictionary[str(row['token_id'])]=token_info(tokenizer,row['token_id'],specials)
    extra=int(bool(rows))
    result.update(final_ids=tokens,answer=answer,predicted_answer=predicted,gold_answer=gold,
                  answer_extraction=method,correct_numeric=predicted==gold and gold is not None,
                  correct_strict=method=='marked' and predicted==gold and gold is not None,
                  correct_lenient=predicted==gold and gold is not None,token_dictionary=dictionary,
                  generation_steps=original['steps'],repair_forwards=extra,steps=original['steps']+extra,
                  repair_elapsed_seconds=repair_seconds,repair_forward_seconds=forward_seconds,
                  elapsed_seconds=original['elapsed_seconds']+repair_seconds)
    # Do not carry a stale word-position map after changing tokens.
    result.pop('word_map',None)
    return result


def repair_window(run, out, threshold=.75):
    """Reuse completed outputs: zero generation reruns, <=1 repair forward/sample."""
    source=Source(Path(run));plans=[];seen=set()
    try:
        for number,name in enumerate(source.names,1):
            original=source.result(name);sid=str(original['sample_id'])
            if sid in seen or Path(sid).name!=sid or sid in ('.','..'):raise ValueError('Invalid/duplicate sample ID')
            seen.add(sid)
            header,candidates=repair_candidates(source.records(name),original,threshold)
            if header['config']['mask_id'] in original['final_ids']:raise ValueError('Original run is incomplete')
            plans.append({'header':header,'original':original,'candidates':candidates})
            if number%10==0 or number==len(source.names):
                print(f'Read repair candidates: {number}/{len(source.names)} questions',flush=True)
    finally:source.close()
    if not plans:raise ValueError('No samples')
    config=plans[0]['header']['config']
    if any(p['header']['config']!=config for p in plans):raise ValueError('Mixed source configs')
    source_digest=hashlib.sha256(json.dumps(plans,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest={'source_run':str(Path(run).resolve()),'source_results_sha256':source_digest,
              'source_config':config,'repair_threshold':threshold,'source_sha256':source_fingerprint(),
              'rule':'all_originally_nonspecial_forced_neighbors_below_threshold_before_original_stop',
              'stop_rule':'original_boundary_fixed; original stop tokens never masked; new special predictions allowed',
              'max_repair_forwards_per_question':1}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        manifest_path=out/'manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())!=manifest:raise ValueError('Repair input/code/settings changed; use fresh --out')
        else:
            if any(p.name!='.lock' for p in out.iterdir()):raise ValueError('Use a new empty --out')
            dump(manifest_path,manifest)
        pending=[p for p in plans if not (out/'samples'/str(p['original']['sample_id'])/'result.json').exists()]
        active=[p for p in pending if p['candidates']]
        model=tokenizer=None
        if active:
            # The only model loading/inference path. Execute this on Vast only.
            model,tokenizer=load_model(config)
        outcomes=[]
        for number,plan in enumerate(plans,1):
            original=plan['original'];sid=str(original['sample_id']);candidates=plan['candidates']
            folder=out/'samples'/sid;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():
                result=json.loads((folder/'result.json').read_text())
            else:
                rows=[];seconds=forward_seconds=0.;tokens=original['final_ids'].copy()
                if candidates:
                    import torch
                    started=time.perf_counter();prompt=plan['header']['prompt_ids'];pl=len(prompt)
                    positions=[c['position'] for c in candidates]
                    for p in positions:tokens[p]=config['mask_id']
                    x=torch.tensor([prompt+tokens],device=model.device)
                    with torch.inference_mode():
                        if x.is_cuda:torch.cuda.synchronize()
                        t0=time.perf_counter();response=model(x,use_cache=False)
                        if x.is_cuda:torch.cuda.synchronize()
                        forward_seconds=time.perf_counter()-t0
                        rows=score_logits(response.logits[0,pl:],positions,{},config['mask_id'],config['top_k'])
                        del response
                        # All predictions are from this single forward. No retries,
                        # threshold gates, confidence acceptance, or sequential fills.
                        for row in rows:tokens[row['position']]=row['token_id']
                    seconds=time.perf_counter()-started
                    result=repaired_result(original,tokens,tokenizer,rows,seconds,forward_seconds)
                else:
                    result=dict(original);result.update(generation_steps=original['steps'],repair_forwards=0,
                                                       repair_elapsed_seconds=0.,repair_forward_seconds=0.)
                changes=[{'position':row['position'],'before_token_id':original['final_ids'][row['position']],
                          'after_token_id':row['token_id'],'after_confidence':row['confidence'],
                          'after_entropy':row['entropy'],'changed':original['final_ids'][row['position']]!=row['token_id']}
                         for row in rows]
                result['repair']={'threshold':threshold,'candidates':candidates,'changes':changes,
                                  'original_answer':original['answer'],'original_correct_numeric':original['correct_numeric'],
                                  'fixed_answer_end':original['answer_token_length']}
                dump(folder/'original_result.json',original)
                dump(folder/'result.json',result)
            outcomes.append((original,result))
            print(f'Repaired {number}/{len(plans)} sample={sid} candidates={len(candidates)} '
                  f'correct={original["correct_numeric"]}->{result["correct_numeric"]} '
                  f'extra_forwards={result["repair_forwards"]}',flush=True)
        flips=Counter('both_correct' if a['correct_numeric'] and b['correct_numeric'] else
                      'repaired_only_correct' if b['correct_numeric'] else
                      'original_only_correct' if a['correct_numeric'] else 'both_wrong' for a,b in outcomes)
        hist=Counter(len(p['candidates']) for p in plans);n=len(plans)
        summary={'samples':n,'repair_threshold':threshold,
                 'original_accuracy':sum(a['correct_numeric'] for a,b in outcomes)/n,
                 'repaired_accuracy':sum(b['correct_numeric'] for a,b in outcomes)/n,
                 'accuracy_change_percentage_points':100*sum(int(b['correct_numeric'])-int(a['correct_numeric']) for a,b in outcomes)/n,
                 'paired_correctness':dict(flips),'candidate_count_per_question':dict(sorted(hist.items())),
                 'total_candidates':sum(k*v for k,v in hist.items()),
                 'extra_repair_forwards':sum(b['repair_forwards'] for a,b in outcomes),
                 'changed_tokens':sum(c['changed'] for a,b in outcomes for c in b['repair']['changes']),
                 'repair_forward_seconds':sum(b['repair_forward_seconds'] for a,b in outcomes),
                 'total_generation_forwards':sum(a['steps'] for a,b in outcomes),
                 'note':'Saved original output reused. One simultaneous repair forward when candidates exist, none otherwise. Original final-stop boundary fixed. No correctness used for candidate selection.'}
        dump(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
        return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--threshold',type=float,default=.75)
    a=p.parse_args();repair_window(a.run,a.out,a.threshold)
