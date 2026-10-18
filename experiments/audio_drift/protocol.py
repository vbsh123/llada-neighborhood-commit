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
