# Pinned harness scoring of saved GSM8K answers

This is a scoring-only adapter. It reads existing `samples/ID/result.json` files,
uses the pinned EleutherAI harness's actual RegexFilter, TakeFirstFilter and
exact-match implementation, and writes a separate report. It never loads a model,
imports torch/datasets, runs generation, or changes prompts, answers or old scores.
Only the already-installed NumPy dependency is needed.

## Provenance and scope

Pinned upstream repository: https://github.com/EleutherAI/lm-evaluation-harness

Commit: `84aa9f95fea2e1bd298e1859cab0b12094f80e0b` (the revision named by ES-dLLM's
official README). Task: `gsm8k`, configuration version 3.0.

`vendor/snapshot.py` contains selected upstream class/function bodies verbatim;
only registration decorators were omitted and imports consolidated. This bundles
the scorer without the full harness's model-generation dependencies. It is not an
installation of the entire harness or an official endorsement of this adapter.
`vendor/gsm8k.yaml` preserves the upstream task configuration; `task.json` is its
JSON serialization, so scoring requires no YAML parser. `provenance.json` records
source file hashes, snippet hashes, commit and bundled file hashes. MIT and
Apache-2.0 licenses and attribution are included. Runtime verifies bundled hashes.

The original configuration's generation prompt/five-shot setting is not applied
to saved text. Our existing question-only zero-shot generation protocol remains
zero-shot. This standardizes extraction and scoring, not the entire experimental
setup. No extra stop strings are retrospectively applied to saved answers.

## Metrics

- Primary: `exact_match,flexible-extract`. Official regex chooses the last numeric
  match from each saved answer, regardless of whether an earlier #### marker exists.
- Secondary: `exact_match,strict-match`. Official regex chooses the first match
  following the exact `#### ` marker format.
- The official metric removes commas, dollar signs, the reference prefix through
  `#### `, and a trailing period; comparison is case-insensitive. It does not use
  Decimal normalization. Therefore `42.0` does not equal `42` under this protocol.
- Missing extraction is `[invalid]` and is counted incorrect. References lacking
  the expected marked numeric format cause an error instead of silently proceeding.

Both metrics score final answers, not reasoning validity. The report retains
legacy correctness and lists cases whose harness score differs, rather than
rewriting historical numbers. A custom parser and these official filters can
disagree in either direction; no score change is assumed to benefit sleep.

## Commands on your existing Vast machine

```bash
cd /workspace/llada-neighborhood-commit
git switch sleep-simulation
git pull --ff-only
source .venv/bin/activate

python -m unittest discover -s tests -p 'test_gsm8k_eval.py' -v

python -m experiments.gsm8k_eval --run runs/real_sleep_smoke_v5
```

No new package installation or model download is required. For the full study,
after generation completes:

```bash
python -m experiments.gsm8k_eval --run runs/real_sleep_study_v1
cat runs/real_sleep_study_v1/gsm8k_harness_eval/summary.json
```

To re-score the saved simulator and real-execution studies in one invocation:

```bash
python -m experiments.gsm8k_eval \
  --run runs/sleep_study_v1 \
  --run runs/real_sleep_study_v1
```

Single saved mode directories work too, e.g. `--run runs/OLD_RUN/top1`. Default
behavior discovers all immediate mode directories containing `samples/`; use
`--modes active_baseline real_sleep` to select modes explicitly. Baseline defaults
to the first available of baseline/active_baseline/top1/original; override with
`--baseline-mode NAME`. Pairing is within a run; unrelated runs are not silently
pooled or compared as if matched.

## Outputs and checks

Each run gets a dedicated `gsm8k_harness_eval/` directory:

- `summary.json`: both accuracies, invalid extractions, disagreements with legacy
  scoring, paired correctness and percentage-point differences.
- `per_sample.jsonl` and `per_sample.csv`: full question/reference/generated answer,
  both extracted predictions and correctness, legacy fields and input hashes.
- `provenance.json`: exact upstream version, scorer hashes and every input result
  file's SHA256. Original result files are read-only inputs.

All modes must have matching completed sample sets, questions and references;
saved `samples.jsonl`, when available, is also checked. This catches accidental
cross-run pairing. Saved-file consistency is not independent verification against
the official dataset: the report explicitly records that distinction.

For an unfinished run, normally wait. `--allow-incomplete` explicitly scores only
the common completed subset and records missing/excluded IDs; it is not a full
study result. The evaluator reads small result JSONs, never large token traces.

Use `--out PATH` for a dedicated alternate report folder with a single `--run`.
Re-running regenerates evaluator reports; generation files and their metrics
remain unchanged. Keep separate report folders if preserving older evaluator
versions. Suggested manual audit: inspect every correctness flip plus a random
sample of agreements and any ambiguous extraction, using the saved CSV.
