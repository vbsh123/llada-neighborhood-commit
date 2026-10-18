"""Prepare reproducible ASR manifests, including streaming test-set samples."""
import argparse
import io
from itertools import zip_longest
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    p.add_argument("--config", default="clean")
    p.add_argument("--split", default="validation")
    p.add_argument("--revision", default="main")
    p.add_argument("--streaming", action="store_true")
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
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    import soundfile as sf

    revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
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
    for row in dataset:
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
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.out.with_suffix(".provenance.json").write_text(json.dumps({
        "dataset": args.dataset, "config": args.config, "split": args.split,
        "revision": revision, "selection": "first eligible per quota after optional seeded shuffle; bins interleaved",
        "streaming": args.streaming, "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer": args.shuffle_buffer if args.streaming else None,
        "max_per_speaker": args.max_per_speaker, "speaker_counts": speaker_counts,
        "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
        "duration_bins": bins, "skipped_duration": skipped_duration,
        "samples": len(rows)}, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
