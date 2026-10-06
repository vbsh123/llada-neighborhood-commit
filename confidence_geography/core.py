"""Pure-Python measurements. All positions are zero-based response token indices."""
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation


def token_category(text, special=False):
    if special:
        return 'special'
    if not text.strip():
        return 'whitespace'
    s = text.strip()
    if re.fullmatch(r'[+−-]?\d[\d,.]*', s):
        return 'number'
    if all(unicodedata.category(c).startswith(('P', 'S')) for c in s):
        return 'punctuation_or_symbol'
    if any(c.isdigit() for c in s):
        return 'mixed_numeric'
    if all(c.isalpha() for c in s):
        return 'alphabetic'
    return 'mixed_text'


def groups(positions):
    result = []
    for p in sorted(set(positions)):
        if result and p == result[-1][-1] + 1:
            result[-1].append(p)
        else:
            result.append([p])
    return result


def nearest(position, anchors):
    if not anchors:
        return {'nearest_previous_commit': None, 'signed_distance': None,
                'distance': None, 'nonlocal': None}
    anchor = min(anchors, key=lambda a: (abs(position - a), a))
    distance = abs(position - anchor)
    return {'nearest_previous_commit': anchor, 'signed_distance': position - anchor,
            'distance': distance, 'nonlocal': distance > 1}


def select(rows, policy, quota, threshold):
    candidates = sorted((r for r in rows if r['eligible']),
                        key=lambda r: (-r['confidence'], r['position']))
    if not candidates:
        return []
    if policy == 'threshold':
        chosen = [r for r in candidates if r['confidence'] >= threshold]
        return chosen or candidates[:1]
    if policy == 'left_to_right':
        return sorted(candidates, key=lambda r: r['position'])[:quota]
    if policy == 'top1':
        return candidates[:1]
    if policy == 'top1_window':
        # Rows represent still-masked slots only. Use the same forward's
        # predictions, with no neighbor-confidence gate or replacement slots.
        anchor = candidates[0]['position']
        by_position = {r['position']: r for r in candidates}
        return [by_position[anchor+offset] for offset in (0, -1, 1)
                if anchor+offset in by_position]
    return candidates[:quota]


def step_metrics(rows, previous, last_commits, thresholds):
    candidates = [r for r in rows if r['eligible']]
    leader = min(candidates, key=lambda r: (-r['confidence'], r['position']))
    global_leader = min(rows, key=lambda r: (-r['confidence'], r['position']))
    p = leader['position']
    frontier = min(r['position'] for r in rows)
    old_leader = previous.get('leader_position') if previous else None
    available = {r['position']: r for r in rows}
    old_remains = old_leader in available if old_leader is not None else None
    m = {'leader_position': p, 'leader_confidence': leader['confidence'],
         'global_leader_position': global_leader['position'],
         'left_frontier': frontier, 'distance_from_frontier': p-frontier,
         'masked_count': len(rows), 'eligible_count': len(candidates),
         'previous_leader': old_leader, 'previous_leader_still_masked': old_remains,
         'leader_changed': p != old_leader if old_leader is not None else None,
         'leader_change_reason': ('initial' if old_leader is None else
                                  'unchanged' if p == old_leader else
                                  'previous_winner_filled' if not old_remains else 'overtook'),
         'leader_delta_confidence': leader.get('delta_confidence'),
         **nearest(p, last_commits)}
    ranked = sorted(candidates, key=lambda r: (-r['confidence'], r['position']))
    m['leader_position_margin'] = (ranked[0]['confidence'] - ranked[1]['confidence']
                                   if len(ranked) > 1 else None)
    local = [r for r in candidates if nearest(r['position'], last_commits)['nonlocal'] is False]
    remote = [r for r in candidates if nearest(r['position'], last_commits)['nonlocal'] is True]
    m['local_candidate_count'] = len(local) if last_commits else None
    m['remote_candidate_count'] = len(remote) if last_commits else None
    m['local_option_available'] = bool(local) if last_commits else None
    m['uniform_choice_nonlocal_rate'] = len(remote)/len(candidates) if last_commits else None
    m['best_local_confidence'] = max((r['confidence'] for r in local), default=None)
    m['best_remote_confidence'] = max((r['confidence'] for r in remote), default=None)
    m['thresholds'] = []
    for scope, population in [('all_masked', rows), ('eligible', candidates),
                               ('eligible_nonspecial', [r for r in candidates if not r['special']])]:
        for t in thresholds:
            above = [r for r in population if r['confidence'] >= t]
            islands = groups(r['position'] for r in above)
            newly = [r for r in above if r.get('previous_confidence') is not None
                     and r['previous_confidence'] < t]
            m['thresholds'].append({
                'scope': scope, 'threshold': t, 'population': len(population),
                'count': len(above), 'fraction': len(above)/len(population) if population else None,
                'islands': [[g[0], g[-1]] for g in islands],
                'island_count': len(islands),
                'nonlocal_count': sum(nearest(r['position'], last_commits)['nonlocal'] is True for r in above)
                                 if last_commits else None,
                'newly_above_count': len(newly),
                'newly_above_nonlocal_count': sum(nearest(r['position'], last_commits)['nonlocal'] is True for r in newly)
                                            if last_commits else None})
    return m


def numeric_answer(text, reference=False):
    # Prefer a marker if one occurs naturally; otherwise use the last numeric string.
    # Dataset references still require a marker. Generated answers do not.
    pattern = r'[-+]?\d[\d,]*(?:\.\d+)?'
    marked = re.findall(r'####\s*(' + pattern + r')', text)
    matches = marked or ([] if reference else re.findall(pattern, text))
    if not matches:
        return None, 'missing'
    try:
        return str(Decimal(matches[-1].replace(',', '')).normalize()), 'marked' if marked else 'last_number'
    except InvalidOperation:
        return None, 'invalid'


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite value in trace')
    if isinstance(value, dict):
        for v in value.values(): finite_json(v)
    if isinstance(value, (list, tuple)):
        for v in value: finite_json(v)
    return value
