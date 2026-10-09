"""Insert observation hooks into pinned upstream SOAR; no decoding edits."""
import ast
import hashlib

COMMIT = 'ec3eb400e41a43dc05db20a49c0219b9a968d28e'
SOURCE_SHA256 = 'a0144a51471e1a265f962c41dc1d8169e466081abfda435a9b2043bc85efce78'


def instrument(source, check_hash=True):
    if check_hash and hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Unexpected upstream SOAR source; refusing to patch')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_soar')
    counts = {'before': 0, 'candidate': 0, 'after': 0}

    class Hooks(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            # Immediately before candidate expansion, after model probabilities.
            if any(isinstance(t, ast.Name) and t.id == 'new_beam_candidates' for t in node.targets):
                counts['before'] += 1
                hook = ast.parse('probe.before(global_step, beam, batch_logits, batch_x0, batch_x0_p, prompt.shape[1], block_length, mask_id, confidence_threshold)').body[0]
                return [hook, node]
            # After all pruning/collapse decisions, before reporting best state.
            if any(isinstance(t, ast.Tuple) and {'best_seq', 'best_score'} <=
                   {x.id for x in t.elts if isinstance(x, ast.Name)} for t in node.targets):
                counts['after'] += 1
                return [ast.parse('probe.after(global_step, new_beam_candidates, uniq_new_beam_candidates, beam)').body[0], node]
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == 'new_beam_candidates' and value.func.attr == 'append'):
                counts['candidate'] += 1
                return [node, ast.parse('probe.candidate(beam_idx, new_beam_candidates[-1])').body[0]]
            return node

    Hooks().visit(function)
    if counts != {'before': 1, 'candidate': 4, 'after': 1}:
        raise ValueError(f'Unexpected SOAR structure: {counts}')
    # Keep helper functions; discard upstream demo, evaluation and model loading.
    keep = {'add_gumbel_noise', 'get_num_transfer_tokens', 'generate_soar'}
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                 or isinstance(n, ast.FunctionDef) and n.name in keep]
    return ast.fix_missing_locations(tree)


def reuse_opportunities(candidates, retained_ids, mask_id, special_ids):
    """One event per extra position relative to best retained candidate.

    Candidate IDs distinguish identical states with different ancestry. Reuse is
    only hypothetical: no token states are merged or verified by a model here.
    """
    by_id = {c['id']: c for c in candidates}
    best = by_id[retained_ids[0]]
    proposals = {}
    for candidate in candidates:
        # Exact-state deduplication does not discard unique prediction work.
        if candidate['id'] in retained_ids or candidate.get('status', 'pruned') != 'pruned':
            continue
        conflicts = [p for p, (a, b) in enumerate(zip(best['state'], candidate['state']))
                     if a != mask_id and b != mask_id and a != b]
        for p, (a, b) in enumerate(zip(best['state'], candidate['state'])):
            if a == mask_id and b != mask_id and b not in special_ids:
                # If another surviving branch already holds it, it was not lost.
                if any(by_id[rid]['state'][p] == b for rid in retained_ids):
                    continue
                entry = proposals.setdefault(p, {'position': p, 'donors': []})
                entry['donors'].append({'candidate_id': candidate['id'], 'token_id': b,
                                        'last_commit': candidate.get('last_commits', {}).get(p),
                                        'state_conflicts_with_best': conflicts})
    for entry in proposals.values():
        tokens = {d['token_id'] for d in entry['donors']}
        entry['donors_agree'] = len(tokens) == 1
        entry['token_id'] = next(iter(tokens)) if len(tokens) == 1 else None
    return best, list(proposals.values())
