"""Prepare reproducible ASR manifests, including streaming test-set samples."""
import argparse
import io
import json
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
    args = p.parse_args()
    if args.samples <= 0 or args.shuffle_buffer < 1 or args.max_per_speaker < 0 or args.max_seconds <= 0:
        raise ValueError("Invalid sample count, shuffle buffer, duration, or speaker cap")
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
        if not len(audio) or len(audio) / rate > args.max_seconds:
            skipped_duration += 1
            continue
        i = len(rows)
        path = audio_dir / f"{i:05d}.wav"
        sf.write(path, audio, rate, subtype="PCM_16")
        rows.append({"id": str(row.get("id", i)), "audio": str(path.relative_to(args.out.parent)),
                     "task": "asr", "prompt": "Transcribe the speech in the audio. Return only the transcription.",
                     "target": row["text"], "metric": "wer", "speaker_id": speaker,
                     "source_dataset": args.dataset, "source_revision": revision})
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        if len(rows) == args.samples:
            break
    if len(rows) != args.samples:
        raise ValueError(f"Only found {len(rows)} eligible clips; use a fresh output path and relax filters")
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.out.with_suffix(".provenance.json").write_text(json.dumps({
        "dataset": args.dataset, "config": args.config, "split": args.split,
        "revision": revision, "selection": "first eligible N after optional seeded shuffle",
        "streaming": args.streaming, "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer": args.shuffle_buffer if args.streaming else None,
        "max_per_speaker": args.max_per_speaker, "speaker_counts": speaker_counts,
        "max_seconds": args.max_seconds, "skipped_duration": skipped_duration,
        "samples": len(rows)}, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
