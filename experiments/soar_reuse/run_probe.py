"""Vast-only SOAR branch-reuse measurement. No merging or extra model forwards."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hooks import COMMIT, SOURCE_SHA256, URL, instrument, reuse_opportunities
from confidence_geography.core import numeric_answer
from confidence_geography.run import dump, emit, load_model, run_lock, token_info, PROMPT_PROTOCOL, SCORING_PROTOCOL


class Probe:
    def __init__(self, handle, special_ids):
        self.handle = handle
        self.special_ids = set(special_ids)
        self.events = []
        self.batch_sizes = []
        self.token_ids = set()

    def before(self, step, beam, logits, predictions, confidence, prompt_length, block_length, mask_id, threshold):
        import torch
        self.pl = prompt_length
        self.mask_id = mask_id
        self.parents = []
        self.candidates = []
        self.candidate_lookup = {}
        self.batch_sizes.append(len(beam))
        for i, (seq, score, block, records) in enumerate(beam):
            state = seq[0, prompt_length:].tolist()
            positions = [p for p, t in enumerate(state)
                         if t == mask_id and block*block_length <= p < (block+1)*block_length]
            rows = {}
            for start in range(0, len(positions), 8):
                pp = positions[start:start+8]
                z = logits[i, [prompt_length+p for p in pp]].float()
                lp = torch.log_softmax(z, dim=-1)
                ent = -(lp.exp()*lp).sum(-1).tolist()
                for p, entropy in zip(pp, ent):
                    rows[p] = {'position': p, 'token_id': int(predictions[i, prompt_length+p]),
                               'confidence': float(confidence[i, prompt_length+p]), 'entropy': entropy}
            self.parents.append({'id': i, 'state': state, 'score': score, 'block': block,
                                 'rows': rows, 'search_trigger': bool(positions) and not any(r['confidence'] > threshold for r in rows.values())})
            self.token_ids.update(state)
            self.token_ids.update(r['token_id'] for r in rows.values())
        # Uniform mixture over evaluated parent branches, only where ALL are
        # masked and eligible. Scores are sums of confidence, not log likelihoods.
        mixture = []
        if len(beam) >= 2:
            common = sorted(set.intersection(*(set(p['rows']) for p in self.parents)))
            for start in range(0, len(common), 8):
                pp = common[start:start+8]
                lp = torch.log_softmax(logits[:, [prompt_length+p for p in pp]].float(), dim=-1)
                prob = lp.exp()
                mean = prob.mean(0)
                average_entropy = -(prob*lp).sum(-1).mean(0)
                mixture_entropy = -(mean*mean.clamp_min(torch.finfo(mean.dtype).tiny).log()).sum(-1)
                for j, p in enumerate(pp):
                    predictions_here = [parent['rows'][p]['token_id'] for parent in self.parents]
                    mixture.append({'position': p, 'top1_agreement': len(set(predictions_here)) == 1,
                                    'token_id': predictions_here[0] if len(set(predictions_here)) == 1 else None,
                                    'mean_branch_entropy': float(average_entropy[j]),
                                    'mixture_entropy': float(mixture_entropy[j]),
                                    'branch_information': max(0., float(mixture_entropy[j]-average_entropy[j]))})
        self.mixture = mixture

    def candidate(self, parent, entry):
        seq, score, block, records = entry
        cid = len(self.candidates)
        self.candidate_lookup[id(entry)] = cid
        commits = [{'position': r['position']-self.pl, 'token_id': r['token_id'],
                    'confidence': r['confidence']} for r in records if r['step'] == len(self.batch_sizes)]
        self.candidates.append({'id': cid, 'parent': parent, 'state': seq[0, self.pl:].tolist(),
                                'score': score, 'block': block, 'commits': commits,
                                'last_commits': {r['position']-self.pl: {'step': r['step']-1, 'confidence': r['confidence']}
                                                 for r in records}})

    def after(self, step, raw, unique, retained):
        retained_ids = [self.candidate_lookup[id(c)] for c in retained]
        unique_ids = {self.candidate_lookup[id(c)] for c in unique}
        for candidate in self.candidates:
            cid = candidate['id']
            candidate['status'] = ('retained' if cid in retained_ids else
                                   'pruned' if cid in unique_ids else 'duplicate')
        best, opportunities = reuse_opportunities(self.candidates, retained_ids, self.mask_id, self.special_ids)
        surviving_rows = self.parents[best['parent']]['rows']
        for opportunity in opportunities:
            row = surviving_rows.get(opportunity['position'])
            opportunity['best_parent_prediction'] = row
            opportunity['best_parent_agrees'] = bool(row and opportunity['donors_agree'] and row['token_id'] == opportunity['token_id'])
        event = {'type': 'step', 'step': step, 'parents': self.parents,
                 'candidates': self.candidates, 'retained_ids': retained_ids,
                 'common_masked_mixture': self.mixture, 'opportunities': opportunities}
        emit(self.handle, event)
        self.events.append({'step': step, 'opportunities': opportunities,
                            'search_parents': sum(p['search_trigger'] for p in self.parents),
                            'beam_width': len(self.parents), 'retained_width': len(retained),
                            'pruned': sum(c['status'] == 'pruned' for c in self.candidates),
                            'duplicates': sum(c['status'] == 'duplicate' for c in self.candidates)})
        if step % 32 == 0:
            print(f'SOAR step={step+1} evaluated_branches={len(self.parents)} retained={len(retained)} reuse_positions={len(opportunities)}', flush=True)

    def summary(self, tokens, answer_end):
        # Retrospective final-match rates are consistency measurements, not
        # correctness labels, and cannot establish causal speed savings.
        opportunities = [o for e in self.events for o in e['opportunities'] if o['position'] < answer_end]
        unanimous = [o for o in opportunities if o['donors_agree']]
        supported = [o for o in unanimous if o['best_parent_agrees']]
        conflict_free = [o for o in supported if any(not d['state_conflicts_with_best'] for d in o['donors'])]
        def counts(rows):
            matched = sum(tokens[o['position']] == o['token_id'] for o in rows)
            return {'proposal_events': len(rows), 'unique_positions': len({o['position'] for o in rows}),
                    'matches_final_output': matched, 'final_match_percent': 100*matched/len(rows) if rows else None}
        return {'model_calls': len(self.batch_sizes), 'evaluated_branch_sequences': sum(self.batch_sizes),
                'beam_width_histogram': dict(Counter(self.batch_sizes)),
                'search_parent_events': sum(e['search_parents'] for e in self.events),
                'steps_with_multiple_parents': sum(e['beam_width'] > 1 for e in self.events),
                'steps_with_pruning': sum(e['pruned'] > 0 for e in self.events),
                'steps_with_reuse_proposals': sum(any(o['position'] < answer_end for o in e['opportunities']) for e in self.events),
                'duplicate_candidates': sum(e['duplicates'] for e in self.events),
                'all_extra_position_events': len(opportunities),
                'donor_consensus': counts(unanimous),
                'also_supported_by_best_parent': counts(supported),
                'also_has_nonconflicting_donor_state': counts(conflict_free),
                'warning': 'Hypothetical reuse, not measured savings. Repeated proposals count as separate events. Final agreement is not ground-truth token correctness.'}


def run(args):
    import torch
    source = Path(args.window)
    original_manifest = json.loads((source/'manifest.json').read_text())
    config = original_manifest['config']
    if config['prompt_protocol'] != PROMPT_PROTOCOL or config['scoring_protocol'] != SCORING_PROTOCOL:
        raise ValueError('Unexpected source prompt/scoring protocol')
    samples = [json.loads(s) for s in (source/'samples.jsonl').read_text().splitlines() if s.strip()][:args.samples]
    if len(samples) != args.samples or args.samples < 1 or args.beam_size < 2:
        raise ValueError('Require available sample count >=1 and beam size >=2')
    length = config['length'];block = config['block_length']
    steps = length  # Official default 128 changed explicitly to window length.
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    manifest = {'upstream_commit': COMMIT, 'upstream_sha256': SOURCE_SHA256,
                'source_window': str(source.resolve()), 'source_config': config,
                'samples_sha256': hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest(),
                'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('hooks.py').read_bytes()).hexdigest(),
                'samples': len(samples), 'beam_size': args.beam_size, 'steps': steps,
                'soar_threshold': .95, 'max_parallel_tokens': 5, 'temperature': 0.,
                'mixture_weights': 'uniform across evaluated parents; common masked eligible positions only',
                'decoding_changes': 'none; upstream policy observed, no merging or extra forwards'}
    with run_lock(out):
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text()) != manifest:
                raise ValueError('Settings/input/code changed; use fresh output')
        elif any(p.name != '.lock' for p in out.iterdir()):
            raise ValueError('Use a new empty output folder')
        else: dump(out/'manifest.json', manifest)
        cache = Path(__file__).parent/'upstream_cache'/f'generate_{COMMIT}.py'
        cache.parent.mkdir(exist_ok=True)
        if not cache.exists():
            print('Downloading pinned upstream SOAR source (not weights)',flush=True)
            with urllib.request.urlopen(URL, timeout=60) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:raise ValueError('Upstream source hash differs')
            cache.write_bytes(raw)
        tree = instrument(cache.read_text())
        pending = [s for s in samples if not (out/'samples'/s['id']/'result.json').exists()]
        if pending:
            model, tokenizer = load_model(config)
            namespace = {}
            exec(compile(tree, str(cache), 'exec'), namespace)
            for sample in pending:
                sid = sample['id'];folder = out/'samples'/sid;folder.mkdir(parents=True,exist_ok=True)
                # Exact original prompt IDs: no new instructions/template choices.
                with gzip.open(source/'samples'/sid/'trace.jsonl.gz', 'rt') as handle:
                    header = json.loads(next(handle))
                if header['type'] != 'header':raise ValueError('Missing original header')
                prompt = torch.tensor([header['prompt_ids']],device=model.device)
                torch.manual_seed(config['seed'])
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                partial = folder/'branches.jsonl.gz.partial'
                started = time.perf_counter()
                with gzip.open(partial,'wt',compresslevel=3) as handle, torch.inference_mode():
                    probe = Probe(handle,tokenizer.all_special_ids)
                    namespace['probe'] = probe
                    emit(handle, {'type': 'header', 'sample': sample, 'prompt_ids': header['prompt_ids'], 'manifest': manifest})
                    class TimedModel:
                        device = model.device
                        seconds = 0.
                        def __call__(self, x):
                            torch.cuda.synchronize();t0 = time.perf_counter()
                            response = model(x, use_cache=False)
                            torch.cuda.synchronize();self.seconds += time.perf_counter()-t0
                            return response
                    timed = TimedModel()
                    tokens = namespace['generate_soar'](timed, prompt, steps=steps, gen_length=length,
                              block_length=block, temperature=0., cfg_scale=0., mask_id=config['mask_id'],
                              max_beam_size=args.beam_size)[0, prompt.shape[1]:].tolist()
                    end = min((i for i,t in enumerate(tokens) if t in header['stop_ids']), default=length)
                    answer = tokenizer.decode(tokens[:end],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    predicted, method = numeric_answer(answer);gold,_ = numeric_answer(sample['answer'],reference=True)
                    result = {'sample_id': sid, 'question': sample['question'], 'reference': sample['answer'],
                              'answer': answer, 'final_ids': tokens, 'answer_token_length': end,
                              'predicted_answer': predicted, 'gold_answer': gold, 'answer_extraction': method,
                              'correct_numeric': predicted == gold and gold is not None,
                              'unresolved_masks': tokens.count(config['mask_id']),
                              'forward_seconds': timed.seconds, 'elapsed_seconds': time.perf_counter()-started,
                              'reuse': probe.summary(tokens,end),
                              'token_dictionary': {str(t): token_info(tokenizer,t,set(tokenizer.all_special_ids))
                                                   for t in sorted(probe.token_ids)}}
                    emit(handle, {'type': 'result', **result})
                partial.replace(folder/'branches.jsonl.gz');dump(folder/'result.json',result)
                print(f'DONE SOAR sample={sid} accuracy={result["correct_numeric"]} calls={result["reuse"]["model_calls"]} reuse_events={result["reuse"]["all_extra_position_events"]}',flush=True)
        results = [json.loads((out/'samples'/s['id']/'result.json').read_text()) for s in samples]
        totals = {'samples': len(results), 'numeric_accuracy': sum(r['correct_numeric'] for r in results)/len(results),
                  'total_model_calls': sum(r['reuse']['model_calls'] for r in results),
                  'total_evaluated_branch_sequences': sum(r['reuse']['evaluated_branch_sequences'] for r in results),
                  'total_forward_seconds': sum(r['forward_seconds'] for r in results),
                  'total_observed_elapsed_seconds': sum(r['elapsed_seconds'] for r in results),
                  'unresolved_mask_count': sum(r['unresolved_masks'] for r in results),
                  'all_extra_position_events': sum(r['reuse']['all_extra_position_events'] for r in results),
                  'steps_with_pruning': sum(r['reuse']['steps_with_pruning'] for r in results),
                  'steps_with_reuse_proposals': sum(r['reuse']['steps_with_reuse_proposals'] for r in results),
                  'steps_with_multiple_parents': sum(r['reuse']['steps_with_multiple_parents'] for r in results),
                  'note': 'Observation-only pilot. No actual merges; no measured merge speedup. Timings include logging overhead, with model time separately measured. Final token agreement is not correctness.'}
        for key in ['donor_consensus','also_supported_by_best_parent','also_has_nonconflicting_donor_state']:
            count = sum(r['reuse'][key]['proposal_events'] for r in results)
            matched = sum(r['reuse'][key]['matches_final_output'] for r in results)
            totals[key] = {'proposal_events': count, 'matches_final_output': matched,
                           'final_match_percent': 100*matched/count if count else None,
                           'questions_with_proposals': sum(r['reuse'][key]['proposal_events'] > 0 for r in results)}
        calls = totals['total_model_calls'];pruning = totals['steps_with_pruning']
        totals['percent_all_steps_with_reuse_proposals'] = 100*totals['steps_with_reuse_proposals']/calls if calls else None
        totals['percent_pruning_steps_with_reuse_proposals'] = 100*totals['steps_with_reuse_proposals']/pruning if pruning else None
        dump(out/'summary.json',totals);print(json.dumps(totals,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window',required=True,help='Completed window run with saved samples/prompts')
    parser.add_argument('--out',required=True)
    parser.add_argument('--samples',type=int,default=10)
    parser.add_argument('--beam-size',type=int,default=2)
    run(parser.parse_args())
