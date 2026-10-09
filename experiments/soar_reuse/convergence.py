"""Same-position cross-branch convergence, following actual retained ancestry."""


def measure_convergence(steps, final_tokens, answer_end, special_ids):
    """One observation per step, ordered donor/recipient pair and position.

    Only positions masked and eligible in both parents, with different top1
    tokens. Repeated observations are retained. A discarded recipient is
    censored, not treated as evidence that its hypothetical future disagrees.
    """
    events = []
    # The returned SOAR output is retained beam[0]. Trace its ancestry backwards.
    winner = {}
    parent_index = 0
    for index in range(len(steps)-1, -1, -1):
        step = steps[index]
        candidate = step['retained'][parent_index]
        winner[index] = candidate['parent']
        parent_index = candidate['parent']
    for index, step in enumerate(steps):
        parents = step['parents']
        for donor in parents:
            for recipient in parents:
                if donor['id'] == recipient['id']:
                    continue
                for p in sorted(set(donor['rows']) & set(recipient['rows'])):
                    a, b = donor['rows'][p], recipient['rows'][p]
                    token = a['token_id']
                    if p >= answer_end or token in special_ids or token == b['token_id']:
                        continue
                    event = {'step': step['step'], 'donor_parent': donor['id'],
                             'recipient_parent': recipient['id'], 'position': p,
                             'token_id': token, 'donor_confidence': a['confidence'],
                             'recipient_support': b['cross_support'][str(token)],
                             'first_top1_delay': None, 'first_commit_delay': None,
                             'last_observed_delay': 0, 'recipient_lineage_extinct': False,
                             'on_final_lineage': winner[index] == recipient['id'],
                             'matches_final_output': None, 'max_later_masked_support': None}
                    descendants = {recipient['id']}
                    for later_index in range(index, len(steps)):
                        later = steps[later_index]
                        delay = later['step']-step['step']
                        event['last_observed_delay'] = delay
                        if later_index > index:
                            support = [later['parents'][j]['rows'][p]['cross_support'][str(token)]
                                       for j in descendants if p in later['parents'][j]['rows']]
                            if support:
                                event['max_later_masked_support'] = max(
                                    support + ([event['max_later_masked_support']]
                                               if event['max_later_masked_support'] is not None else []))
                        if later_index > index and event['first_top1_delay'] is None:
                            if any(later['parents'][j]['rows'].get(p, {}).get('token_id') == token
                                   for j in descendants):
                                event['first_top1_delay'] = delay
                        retained_descendants = {j for j, c in enumerate(later['retained'])
                                                if c['parent'] in descendants}
                        if event['first_commit_delay'] is None and any(
                                later['retained'][j]['state'][p] == token
                                for j in retained_descendants):
                            # Delay zero means revealed in the origin forward.
                            event['first_commit_delay'] = delay
                        descendants = retained_descendants
                        if not descendants:
                            event['recipient_lineage_extinct'] = True
                            break
                    if event['on_final_lineage']:
                        event['matches_final_output'] = final_tokens[p] == token
                    events.append(event)
    def counts(items):
        n = len(items)
        final = [e for e in items if e['on_final_lineage']]
        return {'observations': n,
                'later_top1_count': sum(e['first_top1_delay'] is not None for e in items),
                'retained_descendant_commit_count': sum(e['first_commit_delay'] is not None for e in items),
                'lineage_extinct_count': sum(e['recipient_lineage_extinct'] for e in items),
                'final_lineage_observations': len(final),
                'final_lineage_matches': sum(e['matches_final_output'] for e in final),
                'later_top1_percent': 100*sum(e['first_top1_delay'] is not None for e in items)/n if n else None,
                'retained_descendant_commit_percent': 100*sum(e['first_commit_delay'] is not None for e in items)/n if n else None,
                'final_lineage_match_percent': 100*sum(e['matches_final_output'] for e in final)/len(final) if final else None}
    sweep = []
    for donor_threshold in (.75, .85, .9, .95):
        for support_threshold in (.05, .1, .2, .3, .5, .75, .9):
            selected = [e for e in events if e['donor_confidence'] >= donor_threshold
                        and e['recipient_support'] >= support_threshold]
            sweep.append({'donor_threshold': donor_threshold,
                          'recipient_support_threshold': support_threshold, **counts(selected)})
    return events, {'all_disagreements': counts(events), 'threshold_sweep': sweep,
                    'definition': 'Ordered branch-pair/position/forward observations; same position; any retained descendant for first hits. Extinct lineages are censored. Final matches only for recipients on the actual final winning lineage. Final stop cutoff is retrospective.'}
