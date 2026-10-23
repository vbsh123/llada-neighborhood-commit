"""Prepare a seeded, duration-filtered GSM8K Speech evaluation manifest."""
import argparse
import io
import json
import math
from pathlib import Path
import random
from itertools import zip_longest

from .inputs import canonical_number, read_manifest

DATASET = "mistralai/gsm8k_speech"
PROMPT = ("Solve the math problem spoken in the audio. Think step by step, then end your response "
          "with a separate line: #### <final numerical answer>. "
          "Write only the number after ####, without units or additional text.")


def select_rows(rows, samples, seed, min_seconds, max_seconds, excluded_indices=()):
    """Uniform reservoir over eligible rows; retain only selected audio payloads.

    Consume the local Parquet reader completely and close it before WAV export.
    Selection uses duration and seeded randomness, never model correctness.
    """
    import soundfile as sf
    rng = random.Random(seed)
    selected, eligible, scanned = [], 0, 0
    excluded_indices = set(excluded_indices)
    for index, (source_file, row_index, row) in enumerate(rows):
        scanned += 1
        if index in excluded_indices:
            continue
        payload = row["audio"]
        if payload.get("bytes") is None:
            raise ValueError("Expected embedded audio bytes in GSM8K Speech Parquet")
        info = sf.info(io.BytesIO(payload["bytes"]))
        duration = info.frames / info.samplerate
        if not math.isfinite(duration) or not min_seconds <= duration <= max_seconds:
            continue
        target = canonical_number(str(row["answer"]))
        if target is None:
            raise ValueError(f"Invalid numerical reference at {source_file}:{row_index}")
        eligible += 1
        slot = len(selected) if len(selected) < samples else rng.randrange(eligible)
        if slot < samples:
            item = {"source_file": source_file, "source_row": row_index, "audio_seconds": duration,
                    "target": target, "payload": payload["bytes"], "source_index": index}
            if slot == len(selected):
                selected.append(item)
            else:
                selected[slot] = item
        if scanned % 100 == 0:
            print(f"Scanned {scanned}; eligible {eligible}; retained {len(selected)}/{samples}", flush=True)
    if len(selected) < samples:
        raise ValueError(f"Only {len(selected)} eligible clips; requested {samples}")
    # Reservoir slots give deterministic order; first N run rows stay fixed when
    # the same prepared manifest is used with a smaller --samples argument.
    return selected, {"scanned": scanned, "eligible": eligible}


def parquet_rows(repo, revision, files):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    for name in files:
        print(f"Downloading/caching {name}", flush=True)
        local = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, revision=revision)
        with open(local, "rb") as source:
            with pq.ParquetFile(source) as parquet:
                offset = 0
                for batch in parquet.iter_batches(batch_size=4, use_threads=False):
                    for row in batch.to_pylist():
                        yield name, offset, row
                        offset += 1
        print(f"Closed {name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--revision", default="main")
    p.add_argument("--base-manifest", type=Path, help="Keep these development examples and add disjoint fresh rows")
    p.add_argument('--exclude-manifests',type=Path,nargs='+',default=[],
                   help='Exclude all these source questions; do not copy them into the output')
    p.add_argument('--exclude-id-files',type=Path,nargs='+',default=[],
                   help='JSON files with source_dataset, source_revision and ids; no old WAVs required')
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument('--cohort',help='Explicit cohort label for a dedicated held-out evaluation')
    p.add_argument("--min-seconds", type=float, default=0)
    p.add_argument("--max-seconds", type=float, default=30)
    args = p.parse_args()
    if args.samples < 1 or not 0 <= args.min_seconds < args.max_seconds <= 30:
        raise ValueError("Invalid sample count or duration bounds")
    audio_dir = args.out.parent / (args.out.stem + "_audio")
    if args.out.exists() or audio_dir.exists():
        raise ValueError("Use a fresh manifest path; existing inputs are never overwritten")
    from huggingface_hub import HfApi
    import soundfile as sf
    base = read_manifest(args.base_manifest) if args.base_manifest else []
    excluded_rows=[row for path in args.exclude_manifests for row in read_manifest(path)]
    for path in args.exclude_id_files:
        data=json.loads(path.read_text())
        if (not isinstance(data.get('ids'),list) or not data['ids']
                or not all(isinstance(i,str) for i in data['ids']) or len(set(data['ids']))!=len(data['ids'])):
            raise ValueError('Exclusion ID file must contain unique source question IDs')
        excluded_rows.extend(dict(id=i,source_dataset=data.get('source_dataset'),source_revision=data.get('source_revision')) for i in data['ids'])
    if {r['id'] for r in base}&{r['id'] for r in excluded_rows}:
        raise ValueError('Base examples cannot also be explicitly excluded')
    if base and args.samples <= len(base):
        raise ValueError("Combined sample count must exceed the development manifest size")
    references=base+excluded_rows
    revisions = {r.get('source_revision') for r in references}
    if references and (len(revisions) != 1 or None in revisions or
                 any(r.get('source_dataset') != DATASET for r in references)):
        raise ValueError("Reference manifests must identify a single pinned GSM8K Speech revision")
    if any(not str(r['id']).startswith('gsm8k-speech-') or not str(r['id']).removeprefix('gsm8k-speech-').isdigit() for r in references):
        raise ValueError('Reference manifests must use source-index GSM8K Speech IDs')
    selected_revision = next(iter(revisions)) if references else args.revision
    api = HfApi()
    revision = api.dataset_info(DATASET, revision=selected_revision).sha
    if references and args.revision != 'main' and args.revision != revision:
        raise ValueError("Explicit revision differs from the development dataset revision")
    excluded = {int(r['id'].removeprefix('gsm8k-speech-')) for r in references}
    files = sorted(f for f in api.list_repo_files(DATASET, repo_type="dataset", revision=revision)
                   if f.startswith("data/test-") and f.endswith(".parquet"))
    if not files:
        raise ValueError("No data/test-*.parquet files; inspect dataset layout")
    selected, counts = select_rows(parquet_rows(DATASET, revision, files), args.samples - len(base),
                                   args.seed, args.min_seconds, args.max_seconds, excluded)
    audio_dir.mkdir(parents=True)
    manifest = []
    for row in base:
        # read_manifest resolves and validates the existing audio paths. Keeping
        # absolute paths avoids copying or altering the development WAV files.
        manifest.append({**row, 'cohort': 'development'})
    for i, row in enumerate(selected):
        audio, rate = sf.read(io.BytesIO(row["payload"]), dtype="float32")
        path = audio_dir / f"{i:05}.wav"
        sf.write(path, audio, rate, subtype="PCM_16")
        manifest.append({"id": f"gsm8k-speech-{row['source_index']:05}",
                         "audio": str(path.relative_to(args.out.parent)), "task": "gsm8k_speech",
                         "prompt": PROMPT, "metric": "numeric", "target": row["target"],
                         "audio_seconds": row["audio_seconds"], "cohort": args.cohort or ("fresh" if base else ('validation_v21' if excluded_rows else 'unspecified')),
                         "source_dataset": DATASET,
                         "source_revision": revision, "source_file": row["source_file"],
                         "source_row": row["source_row"]})
        print(f"Selected {i+1}/{args.samples}: {manifest[-1]['id']} ({row['audio_seconds']:.2f}s)", flush=True)
    if base:
        development, fresh = manifest[:len(base)], manifest[len(base):]
        manifest = [r for pair in zip_longest(development, fresh) for r in pair if r is not None]
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row) + "\n" for row in manifest))
    temp.replace(args.out)
    args.out.with_suffix(".provenance.json").write_text(json.dumps({
        "dataset": DATASET, "revision": revision, "source_files": files, "seed": args.seed,
        "selection": "uniform reservoir over duration-eligible test rows; no correctness filtering",
        "samples": args.samples, "development_samples": len(base), "fresh_samples": len(selected) if references else 0,
        "base_manifest": str(args.base_manifest.resolve()) if args.base_manifest else None,
        "exclude_manifests": [str(p.resolve()) for p in args.exclude_manifests],
        "exclude_id_files": [str(p.resolve()) for p in args.exclude_id_files],
        "excluded_source_indices": sorted(excluded), "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
        "prompt": PROMPT, "scoring": "exact canonical numerical match; explicit final marker or bare number",
        **counts}, indent=2) + "\n")
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
