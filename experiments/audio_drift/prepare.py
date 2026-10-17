"""Download a tiny real-speech ASR smoke dataset and write an explicit manifest."""
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
    args = p.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.out.exists():
        raise ValueError("Use a fresh manifest path; existing inputs are never overwritten")
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi
    import soundfile as sf

    revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    dataset = load_dataset(args.dataset, args.config, split=args.split, revision=revision)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if args.samples > len(dataset):
        raise ValueError(f"Requested {args.samples} examples; dataset has {len(dataset)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out.parent / (args.out.stem + "_audio")
    audio_dir.mkdir(exist_ok=False)
    rows = []
    for i in range(args.samples):
        row = dataset[i]
        payload = row["audio"]
        source = io.BytesIO(payload["bytes"]) if payload.get("bytes") is not None else payload["path"]
        audio, rate = sf.read(source, dtype="float32")
        path = audio_dir / f"{i:05d}.wav"
        sf.write(path, audio, rate, subtype="PCM_16")
        rows.append({"id": str(row.get("id", i)), "audio": str(path.relative_to(args.out.parent)),
                     "task": "asr", "prompt": "Transcribe the speech in the audio. Return only the transcription.",
                     "target": row["text"], "metric": "wer"})
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.out.with_suffix(".provenance.json").write_text(json.dumps({
        "dataset": args.dataset, "config": args.config, "split": args.split,
        "revision": revision, "selection": "first N in dataset order", "samples": len(rows)}, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
