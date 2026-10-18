"""Prepare reproducible ASR manifests, including streaming test-set samples."""
import argparse
import io
from itertools import zip_longest
import json
import math
from pathlib import Path
import random


def local_parquet_rows(repo, config, split, revision, api):
    """Read only local split shards, without an abandoned Arrow Dataset scanner.

    HF datasets/Arrow partially consumed streaming scanners can hang or abort
    at interpreter finalization: https://github.com/apache/arrow/issues/45214.
    This path downloads only the requested split, reads it completely using
    ParquetFile, and closes each file before selection starts.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    prefix = f"{config}/{split}/"
    files = sorted(p for p in api.list_repo_files(repo, repo_type="dataset", revision=revision)
                   if p.startswith(prefix) and p.endswith(".parquet"))
    if not files:
        raise ValueError(f"No Parquet shards under {repo}/{prefix}; use the datasets reader for other layouts")
    rows = []
    for name in files:
        print(f"Downloading/caching split shard: {name}", flush=True)
        local = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", revision=revision)
        with open(local, "rb") as source:
            with pq.ParquetFile(source) as parquet:
                for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                    rows.extend(batch.to_pylist())
        print(f"Read {len(rows)} split rows; shard closed", flush=True)
    return rows, files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    p.add_argument("--config", default="clean")
    p.add_argument("--split", default="validation")
    p.add_argument("--revision", default="main")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--reader", choices=["datasets", "local-parquet"], default="datasets")
    p.add_argument("--shuffle-seed", type=int)
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--max-per-speaker", type=int, default=0,
                   help="0 is unlimited; positive values limit clips per speaker")
    p.add_argument("--max-seconds", type=float, default=30)
    p.add_argument("--min-seconds", type=float, default=0)
    p.add_argument("--duration-bins", nargs="+", default=[], metavar="MIN:MAX:COUNT",
                   help="Optional disjoint duration quotas; counts must sum to --samples")
    args = p.parse_args()
    if (args.samples <= 0 or args.shuffle_buffer < 1 or args.max_per_speaker < 0
            or not 0 <= args.min_seconds < args.max_seconds <= 30):
        raise ValueError("Invalid sample count, shuffle buffer, duration, or speaker cap")
    bins = []
    for spec in args.duration_bins:
        low, high, count = spec.split(":")
        low, high, count = float(low), float(high), int(count)
        if not args.min_seconds <= low < high <= args.max_seconds or count < 1:
            raise ValueError("Invalid duration bin")
        bins.append({"min_seconds": low, "max_seconds": high, "requested": count, "selected": 0})
    bins.sort(key=lambda b: b["min_seconds"])
    if bins and (sum(b["requested"] for b in bins) != args.samples or
                 any(a["max_seconds"] > b["min_seconds"] for a, b in zip(bins, bins[1:]))):
        raise ValueError("Duration bins must be disjoint and their counts must sum to --samples")
    if args.out.exists():
        raise ValueError("Use a fresh manifest path; existing inputs are never overwritten")
    if args.reader == "local-parquet" and args.streaming:
        raise ValueError("local-parquet downloads only split shards and cannot be combined with --streaming")
    from huggingface_hub import HfApi
    import soundfile as sf

    api = HfApi()
    print(f"Resolving {args.dataset} / {args.config} / {args.split}", flush=True)
    revision = api.dataset_info(args.dataset, revision=args.revision).sha
    source_files = []
    if args.reader == "local-parquet":
        dataset, source_files = local_parquet_rows(args.dataset, args.config, args.split, revision, api)
        if args.shuffle_seed is not None:
            random.Random(args.shuffle_seed).shuffle(dataset)
    else:
        from datasets import Audio, load_dataset
        dataset = load_dataset(args.dataset, args.config, split=args.split, revision=revision,
                               streaming=args.streaming)
        dataset = dataset.cast_column("audio", Audio(decode=False))
        if args.shuffle_seed is not None:
            options = {"buffer_size": args.shuffle_buffer} if args.streaming else {}
            dataset = dataset.shuffle(seed=args.shuffle_seed, **options)
    if not args.streaming and args.samples > len(dataset):
        raise ValueError(f"Requested {args.samples} examples; dataset has {len(dataset)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out.parent / (args.out.stem + "_audio")
    audio_dir.mkdir(exist_ok=False)
    rows = []
    speaker_counts = {}
    skipped_duration = 0
    for scanned, row in enumerate(dataset, 1):
        if scanned == 1 or scanned % 100 == 0:
            print(f"Scanned {scanned} rows; selected {len(rows)}/{args.samples}", flush=True)
        speaker = str(row.get("speaker_id", "unknown"))
        if args.max_per_speaker and speaker == "unknown":
            raise ValueError("Speaker cap requested, but the dataset has no speaker_id")
        if args.max_per_speaker and speaker_counts.get(speaker, 0) >= args.max_per_speaker:
            continue
        payload = row["audio"]
        source = io.BytesIO(payload["bytes"]) if payload.get("bytes") is not None else payload["path"]
        audio, rate = sf.read(source, dtype="float32")
        duration = len(audio) / rate
        if not len(audio) or not math.isfinite(duration) or not args.min_seconds <= duration <= args.max_seconds:
            skipped_duration += 1
            continue
        duration_bin = None
        if bins:
            # Half-open bins, except the final upper endpoint is included.
            duration_bin = next((b for b in bins if b["min_seconds"] <= duration and
                                 (duration < b["max_seconds"] or
                                  b is bins[-1] and duration == b["max_seconds"])), None)
            if duration_bin is None or duration_bin["selected"] >= duration_bin["requested"]:
                skipped_duration += 1
                continue
        i = len(rows)
        path = audio_dir / f"{i:05d}.wav"
        sf.write(path, audio, rate, subtype="PCM_16")
        rows.append({"id": str(row.get("id", i)), "audio": str(path.relative_to(args.out.parent)),
                     "task": "asr", "prompt": "Transcribe the speech in the audio. Return only the transcription.",
                     "target": row["text"], "metric": "wer", "speaker_id": speaker,
                     "source_dataset": args.dataset, "source_revision": revision,
                     "audio_seconds": duration,
                     "duration_bin": (f"{duration_bin['min_seconds']:g}-{duration_bin['max_seconds']:g}"
                                      if duration_bin else None)})
        if duration_bin:
            duration_bin["selected"] += 1
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        print(f"Selected {len(rows)}/{args.samples}: {rows[-1]['id']} ({duration:.2f}s); "
              f"quotas={[(b['selected'], b['requested']) for b in bins]}", flush=True)
        if len(rows) == args.samples:
            break
    if len(rows) != args.samples:
        raise ValueError(f"Only found {len(rows)} eligible clips; use a fresh output path and relax filters")
    if bins:
        # Interleave strata so a smaller --samples run and the first profiles
        # cover both durations. Audio file paths remain attached to each row.
        buckets = [[r for r in rows if r["duration_bin"] == f"{b['min_seconds']:g}-{b['max_seconds']:g}"]
                   for b in bins]
        rows = [r for group in zip_longest(*buckets) for r in group if r is not None]
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(args.out)
    args.out.with_suffix(".provenance.json").write_text(json.dumps({
        "dataset": args.dataset, "config": args.config, "split": args.split,
        "revision": revision, "selection": "first eligible per quota after optional seeded shuffle; bins interleaved",
        "reader": args.reader, "source_files": source_files,
        "streaming": args.streaming, "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer": args.shuffle_buffer if args.streaming else None,
        "max_per_speaker": args.max_per_speaker, "speaker_counts": speaker_counts,
        "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
        "duration_bins": bins, "skipped_duration": skipped_duration,
        "samples": len(rows)}, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
