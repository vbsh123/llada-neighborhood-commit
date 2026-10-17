"""Explicit task manifest, audio validation, and simple reference scoring."""
import hashlib
import json
from pathlib import Path
import re


def read_manifest(path, samples=None):
    path = Path(path).resolve()
    content = path.read_text()
    rows = json.loads(content) if content.lstrip().startswith("[") else [
        json.loads(line) for line in content.splitlines() if line.strip()]
    result, seen = [], set()
    for row in rows:
        if not all(row.get(k) for k in ("id", "audio", "task", "prompt")):
            raise ValueError("Each manifest row needs id, audio, task, prompt")
        row = dict(row)
        row["id"] = str(row["id"])
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        seen.add(row["id"])
        row["audio"] = str((path.parent / row["audio"]).resolve())
        if not Path(row["audio"]).is_file():
            raise ValueError(f"Missing audio: {row['audio']}")
        if row.get("metric", "none") not in {"none", "wer", "exact", "mcq"}:
            raise ValueError("metric must be none, wer, exact, or mcq")
        if row.get("metric", "none") != "none" and not isinstance(row.get("target"), str):
            raise ValueError("Scored manifest rows need a string target")
        row["audio_sha256"] = hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest()
        result.append(row)
    if samples is not None:
        if samples <= 0:
            raise ValueError("samples must be positive")
        result = result[:samples]
    if not result:
        raise ValueError("Empty manifest")
    return result


def normalized(text):
    return re.findall(r"\w+", text.casefold())


def score(text, target, metric):
    if metric == "none":
        return None
    if metric == "exact":
        return {"metric": metric, "correct": normalized(text) == normalized(target)}
    if metric == "mcq":
        # Deliberately conservative: A, (A), A., or 'Answer: A', not a letter
        # extracted from an arbitrary reasoning paragraph.
        match = re.match(r"^\s*(?:answer\s*:\s*)?\(?([A-Z])\)?(?:[.\s:]|$)", text, re.I)
        prediction = match.group(1).upper() if match else None
        return {"metric": metric, "prediction": prediction,
                "correct": prediction == target.strip().upper()}
    if metric != "wer":
        raise ValueError(metric)
    reference, hypothesis = normalized(target), normalized(text)
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, prediction in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (word != prediction)))
        previous = current
    edits = previous[-1]
    return {"metric": metric, "edits": edits, "reference_words": len(reference),
            "wer": edits / max(1, len(reference))}


def load_audio(path, max_seconds=30):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    array, rate = sf.read(path, dtype="float32", always_2d=True)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Audio is empty or non-finite")
    duration = len(array) / rate
    if duration > max_seconds:
        raise ValueError(f"Audio is {duration:.2f}s; limit is {max_seconds}s. Segment it explicitly.")
    wave = torch.from_numpy(array.mean(axis=1))
    if rate != 16000:
        wave = torchaudio.functional.resample(wave, rate, 16000)
    return wave.numpy(), duration
