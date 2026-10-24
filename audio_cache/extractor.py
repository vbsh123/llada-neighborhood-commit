"""Reference-blind secondary final-answer extraction; preserve original scores.

Accept explicit final labels or an unambiguous terminal conclusion. Never search
for the reference number or choose a number from arbitrary intermediate work.
This grades final numerical answers, not full reasoning consistency.
"""
import re

from .scoring import canonical_number,final_number

VERSION='explicit-conclusion-v1'
NUMBER=r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
UNITS=r'(?:dollars?|cents?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|miles?|feet|inches|grams?|kilograms?|books?|bookcases?|marbles?|letters?|tiles?|cand(?:y|ies)|candy bars?|items?|people|students?|percent|%)'


def scalar_with_units(text):
    text=text.strip().rstrip('.!').strip()
    m=re.fullmatch(r'\$?\s*('+NUMBER+r')(?:\s*'+UNITS+r')?(?:\s+(?:left|remaining|combined|early|in total))?',text,re.I)
    return canonical_number(m.group(1)) if m else None


def terminal_value(text,explicit=False):
    text=text.strip()
    if not text or '?' in text or re.search(r'\bor\b',text,re.I):return None
    scalar=scalar_with_units(text)
    if scalar is not None:return scalar
    # An explicit final section can contain a single answer sentence. Otherwise
    # require a genuine conclusion cue, not the last number in arbitrary prose.
    cue=re.search(r'\b(?:so|therefore|thus|hence)\b',text,re.I)
    if cue:text=text[cue.end():].strip(' ,:')
    elif not explicit and not re.search(r'\b(?:will (?:be|need|take)|final answer is|answer is)\b',text,re.I):
        return None
    if '=' in text:
        scalar=scalar_with_units(text.rsplit('=',1)[-1])
        if scalar is not None:return scalar
    future=re.search(r'\bwill\s+(?:need|be)\s+(?:about\s+|approximately\s+)?\$?\s*('+NUMBER+r')\s*'+UNITS+r'\b',text,re.I)
    if not future:
        future=re.search(r'\bwill\s+take\s+(?:[A-Za-z\'-]+\s+){0,4}('+NUMBER+r')\s*'+UNITS+r'\b',text,re.I)
    if future:return canonical_number(future.group(1))
    matches=re.findall(r'(?<![\w.])('+NUMBER+r')(?!\w|\.\d)',text)
    return canonical_number(matches[0]) if len(matches)==1 else None


def extract_answer(text):
    parsed=final_number(text)
    if parsed is not None:return parsed,'original_marker_or_scalar'
    clean=text.replace('**','').replace('__','')
    for token in ('\\(','\\)','\\[','\\]'):clean=clean.replace(token,'')
    markers=list(re.finditer(r'#{3,}|\bfinal\s+(?:numerical\s+)?answer\s*:',clean,re.I))
    if markers:
        suffix=clean[markers[-1].end():].strip()
        suffix=re.sub(r'^(?:<\s*)?(?:(?:final\s+)?numerical\s+answer|final\s+answer|answer)(?:\s*>)?\s*[:,]?\s*','',suffix,flags=re.I)
        if suffix:
            prediction=terminal_value(suffix,explicit=True)
            return prediction,'expanded_final_marker' if prediction is not None else 'ambiguous_final_marker'
        # Empty trailing marker: inspect the last actual conclusion, not an
        # earlier operand selected by reference matching.
        clean=clean[:markers[-1].start()].rstrip()
    lines=[line.strip() for line in clean.splitlines() if line.strip()]
    if not lines:return None,'no_answer'
    prediction=terminal_value(lines[-1])
    return prediction,'terminal_conclusion' if prediction is not None else 'no_unambiguous_conclusion'
