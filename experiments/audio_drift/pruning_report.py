"""Activation, removal budgets and paired quality/latency for delayed pruning."""
import argparse
from collections import defaultdict
from pathlib import Path

from .diagnostic_report import Archive, write_csv


def analyze(run, out=None):
    archive = Archive(run)
    if out is None and archive.zip:
        archive.close()
        raise ValueError('ZIP analysis requires --out')
    out = Path(out) if out else Path(run)/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        names = set(archive.names())
        for name in sorted(names):
            if not name.startswith('samples/') or not name.endswith('/result.json') or Path(name).parts[1].startswith('.'):
                continue
            r = archive.read(name); bench = r.get('benchmark', {})
            folder = name.rsplit('/',1)[0]
            replay_path = folder+'/fixed_input_replay.json'
            replay = archive.read(replay_path) if replay_path in names else {}
            replays = {s['condition']: s for s in replay.get('policies', [])}
            for policy in bench.get('policies', []):
                spec = policy.get('pruning')
                if not spec: continue
                account = policy['accounting'][0]
                picks = account['pruning']['selections']
                pick = picks[0] if picks else {}
                same = replays.get(policy['condition'], {})
                rows.append(dict(id=r['id'], condition=policy['condition'], scope=spec['scope'],
                    selection=spec['selection'], start=spec['start'], layer_start=spec['layer_start'],
                    cache_interval=policy['schedule'][0]['refresh_every'],
                    requested_audio_fraction=spec['fraction'], baseline_correct=r['score'].get('correct'),
                    policy_correct=policy['score'].get('correct'), parse_failed=policy['score'].get('parse_failed'),
                    baseline_forwards=bench['baseline_forwards'], policy_forwards=account['forwards'],
                    activated=bool(picks), activated_layers=account['pruning']['activated_layers'],
                    active_pruning_forwards=max(0,account['forwards']-spec['start']) if picks else 0,
                    removed_tokens=pick.get('drop_budget',0), removed_acoustic=pick.get('dropped_by_group',{}).get('acoustic',0),
                    removed_semantic=pick.get('dropped_by_group',{}).get('semantic',0),
                    removed_prompt=pick.get('dropped_by_group',{}).get('prompt',0),
                    actual_scope_fraction=pick.get('actual_scope_fraction'), removed_attention_mass=pick.get('removed_attention_mass'),
                    baseline_decode_seconds=bench['baseline_median_seconds'], policy_decode_seconds=policy['median_seconds'],
                    replay_baseline_seconds=replay.get('baseline_median_seconds'), replay_policy_seconds=same.get('median_seconds'),
                    replay_forwards=replay.get('forwards'),
                    replay_activated=bool(same.get('accounting',[{}])[0].get('pruning',{}).get('selections')),
                    computed_token_layers=account['computed_token_layers'],
                    qk_query_key_pairs=sum(account['qk_query_key_pairs_by_layer'])))
        write_csv(out/'pruning_pairs.csv', rows)
        grouped=defaultdict(list)
        for row in rows:grouped[row['condition']].append(row)
        lines=['# Delayed pruning pilot', '',
            'All primary quality results retain baseline failures and examples ending before the pruning anchor. '
            'Activated subsets are diagnostics, not replacement headline scores. Count-matched scope controls '
            'remove different fractions of each group. Timings include one-shot selection and key/value gathering. '
            'Equal-input replay is separate from autonomous decoding; its selections can differ from autonomous selections.', '',
            '| Policy | N | Activated | Baseline correct | Policy correct | Correct → wrong | Wrong → correct | Decode speedup | Replay speedup |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name, v in grouped.items():
            replay=[r for r in v if r['replay_policy_seconds'] is not None]
            speed=sum(r['baseline_decode_seconds'] for r in v)/sum(r['policy_decode_seconds'] for r in v)
            replay_speed=sum(r['replay_baseline_seconds'] for r in replay)/sum(r['replay_policy_seconds'] for r in replay) if replay else None
            lines.append(f"| {name} | {len(v)} | {sum(r['activated'] for r in v)} | {sum(bool(r['baseline_correct']) for r in v)} | "
                f"{sum(bool(r['policy_correct']) for r in v)} | {sum(r['baseline_correct'] and not r['policy_correct'] for r in v)} | "
                f"{sum(not r['baseline_correct'] and r['policy_correct'] for r in v)} | {speed:.3f}× | "
                +(f'{replay_speed:.3f}×' if replay_speed else '—')+' |')
        (out/'pruning_report.md').write_text('\n'.join(lines)+'\n')
        print(f'Wrote delayed pruning report: {len(grouped)} conditions, {len(rows)} paired results',flush=True)
        return rows
    finally:archive.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path)
    args=parser.parse_args();analyze(args.run,args.out)
