"""Explicit, reproducible conditions for the audio-group timing study."""


def conditions(modes, groups=("both",), refresh_intervals=(), mixed_refresh=(), triple_refresh=()):
    if len(set(modes)) != len(modes) or not set(modes) <= {"freeze", "remove", "text_only"}:
        raise ValueError("Intervention modes must be unique: freeze, remove, text_only")
    if not groups or not set(groups) <= {"acoustic", "semantic", "both", "prompt", "prefix", "audio_matched"}:
        raise ValueError("Invalid freeze groups")
    if any(r < 1 for r in refresh_intervals):
        raise ValueError("Refresh intervals must be positive (permanent freeze is always included)")
    mixed = []
    for pair in mixed_refresh:
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError("Mixed refresh must be ACOUSTIC:SEMANTIC, e.g. 8:2")
        a, s = map(int, parts)
        if min(a, s) < 0:
            raise ValueError("Mixed refresh intervals must be nonnegative; 0 means never refresh")
        mixed.append((a, s))
    triples = []
    for item in triple_refresh:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError("Triple refresh must be ACOUSTIC:PROMPT:SEMANTIC")
        triple = tuple(map(int, parts))
        if min(triple) < 0:
            raise ValueError("Triple refresh intervals must be nonnegative")
        triples.append(triple)
    if (refresh_intervals or mixed_refresh or triple_refresh) and "freeze" not in modes:
        raise ValueError("Refresh schedules require --interventions freeze")
    result = []
    for mode in modes:
        if mode != "freeze":
            result.append({"condition": mode, "mode": mode, "schedule": []})
            continue
        for group in dict.fromkeys(groups):
            for interval in [0, *sorted(set(refresh_intervals))]:
                name = "freeze" if group == "both" and interval == 0 else f"freeze_{group}"
                if interval:
                    name += f"_r{interval}"
                result.append({"condition": name, "mode": mode,
                               "schedule": [{"group": group, "refresh_every": interval}]})
        for a, s in dict.fromkeys(mixed):
            result.append({"condition": f"freeze_mixed_a{a}_s{s}", "mode": mode,
                           "schedule": [{"group": "acoustic", "refresh_every": a},
                                        {"group": "semantic", "refresh_every": s}]})
        for a, p, s in dict.fromkeys(triples):
            result.append({"condition": f"freeze_triple_a{a}_p{p}_s{s}", "mode": mode,
                           "schedule": [{"group": "acoustic", "refresh_every": a},
                                        {"group": "prompt", "refresh_every": p},
                                        {"group": "semantic", "refresh_every": s}]})
    return result


def anchors_for(steps, fractions):
    if steps < 1 or not fractions or not all(0 <= f < 1 for f in fractions):
        raise ValueError("Branch fractions must be in [0, 1)")
    return sorted({round(f * (steps - 1)) for f in fractions if round(f * (steps - 1)) + 1 < steps})


def work_estimate(steps, fractions, specifications):
    anchors = anchors_for(steps, fractions) if specifications else []
    return {"baseline_forwards": steps, "branches": len(anchors) * len(specifications),
            "continuation_forwards": len(specifications) * sum(steps - a - 1 for a in anchors),
            "anchor_replay_forwards": 2 * len(anchors),
            "note": "Excludes official-decoder verification. All conditions run sequentially."}


def with_depth_variants(specifications, multipliers):
    """Pair every non-identity flat policy with the same depth-factor control."""
    if not multipliers or any(not isinstance(m, int) or m < 1 for m in multipliers):
        raise ValueError("Depth multipliers must be positive integers")
    if any(a % b for a, b in zip(multipliers, multipliers[1:])):
        raise ValueError("Shallow depth factors must be integer multiples of deeper factors")
    result = []
    suffix = "_depth" + "x".join(map(str, multipliers))
    for spec in specifications:
        result.append(spec)
        if spec['mode'] != 'freeze' or is_identity(spec):
            continue
        result.append({**spec, 'condition': spec['condition'] + suffix,
                       'flat_condition': spec['condition'], 'depth_multipliers': list(multipliers),
                       'schedule': [{**s, 'refresh_by_layer_group': [s['refresh_every'] * m for m in multipliers]}
                                    for s in spec['schedule']]})
    return result


def layer_intervals(specification, num_layers):
    """Expand equal-sized layer groups into an explicit interval per block."""
    grouped = specification.get('refresh_by_layer_group')
    if grouped is None:
        interval = specification['refresh_every']
        if not isinstance(interval, int) or interval < 0:
            raise ValueError("Refresh interval must be a nonnegative integer")
        return (interval,) * num_layers
    if (not grouped or num_layers % len(grouped) or
            any(not isinstance(r, int) or r < 0 for r in grouped)):
        raise ValueError("Layer groups must divide the model depth and have nonnegative integer intervals")
    if any(a and (not b or a % b) for a, b in zip(grouped, grouped[1:])):
        raise ValueError("Shallow refresh intervals must be integer multiples of deeper intervals")
    width = num_layers // len(grouped)
    return tuple(r for r in grouped for _ in range(width))


def is_identity(specification):
    return (not specification.get('pruning') and specification['mode'] == 'freeze' and bool(specification['schedule']) and
            all(all(r == 1 for r in s.get('refresh_by_layer_group', [s['refresh_every']]))
                for s in specification['schedule']))
