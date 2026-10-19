"""Explicit task manifest, audio validation, and simple reference scoring."""
from decimal import Decimal, InvalidOperation
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
        if row.get("metric", "none") not in {"none", "wer", "exact", "mcq", "numeric"}:
            raise ValueError("metric must be none, wer, exact, mcq, or numeric")
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


# Accept one finite scalar, with optional currency and valid thousands grouping.
_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def canonical_number(value):
    value = value.strip()
    match = re.fullmatch(r"\$?\s*(" + _NUMBER + r")", value)
    if not match:
        return None
    try:
        number = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    # Avoid Decimal.normalize(), whose rounding depends on the global context.
    result = format(number, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return "0" if number == 0 else result


def final_number(text, normalize_labels=True):
    # An explicit final marker wins; malformed marked answers fail rather than
    # falling back to an earlier intermediate calculation. Never take last number
    # from arbitrary reasoning, where it may be an operand rather than an answer.
    markers = list(re.finditer(r"####|final\s+answer\s*:", text, re.I))
    answer = text[markers[-1].end():].strip() if markers else text.strip()
    if markers and normalize_labels:
        answer = re.sub(r"^(?:<final numerical answer>\s*|answer\s*:\s*)", "", answer, flags=re.I)
        if answer.startswith(','):
            answer = answer[1:].strip()
    answer = answer.rstrip(".").strip()
    return canonical_number(answer)


def score(text, target, metric):
    if metric == "none":
        return None
    if metric == "numeric":
        reference = canonical_number(target)
        if reference is None:
            raise ValueError("Numeric targets must be a finite numeric scalar")
        prediction = final_number(text)
        strict = final_number(text, normalize_labels=False)
        return {"metric": metric, "prediction": prediction, "reference": reference,
                "scoring_version": "numeric-labels-comma-v3", "strict_prediction": strict,
                "strict_correct": strict == reference, "format_normalized": prediction != strict,
                "parse_failed": prediction is None, "correct": prediction == reference}
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
