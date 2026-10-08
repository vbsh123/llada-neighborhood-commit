"""Run ONLY top1 on the exact saved questions/config of a completed window run."""
import argparse
import hashlib
import json
from pathlib import Path

from .run import execute, run_lock, PROMPT_PROTOCOL, SCORING_PROTOCOL


def matched_inputs(window):
    window = Path(window)
    manifest = json.loads((window / 'manifest.json').read_text())
    config = dict(manifest['config'])
    if config['policy'] != 'top1_window':
        raise ValueError('Expected a top1_window source run')
    if (config['prompt_protocol'], config['scoring_protocol']) != (PROMPT_PROTOCOL, SCORING_PROTOCOL):
        raise ValueError('Source prompt/scoring protocol differs from current collector')
    raw = (window / 'samples.jsonl').read_bytes()
    samples = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(samples) != config['samples']:
        raise ValueError('Saved sample count differs from source config')
    seen = set()
    for sample in samples:
        sid = str(sample['id'])
        if sid in seen or Path(sid).name != sid or sid in ('', '.', '..'):
            raise ValueError('Invalid or duplicate sample ID')
        seen.add(sid)
        folder = window / 'samples' / sid
        result = json.loads((folder / 'result.json').read_text())
        if not (folder / 'trace.jsonl.gz').exists():
            raise ValueError('Source window run is incomplete')
        if (str(result['sample_id']), result['question'], result['reference']) != (sid, sample['question'], sample['answer']):
            raise ValueError('Source question/reference mismatch')
    config['policy'] = 'top1'
    provenance = {**manifest['dataset_info'], 'matched_source_run': str(window.resolve()),
                  'matched_samples_sha256': hashlib.sha256(raw).hexdigest()}
    return config, samples, provenance


def run_matched_baseline(window, out):
    config, samples, provenance = matched_inputs(window)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'manifest.json').exists() and any(p.name != '.lock' for p in out.iterdir()):
        raise ValueError('Use a new empty baseline output folder')
    if (out / 'manifest.json').exists():
        saved = [json.loads(line) for line in (out / 'samples.jsonl').read_text().splitlines() if line.strip()]
        if saved != samples:
            raise ValueError('Baseline sample list differs; use a new output folder')
    print(f'Running ONLY top1 on {len(samples)} saved questions, preserving IDs and order.', flush=True)
    with run_lock(out):
        execute(config, out, matched_samples=samples, matched_dataset_info=provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run_matched_baseline(args.window, args.out)
