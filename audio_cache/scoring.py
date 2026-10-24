"""Unchanged original numerical scorer, for comparison with prior runs."""
from decimal import Decimal, InvalidOperation
import re

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


def score(text,target):
    metric = "numeric"
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
