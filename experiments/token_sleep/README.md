# Simulated sleep of unresolved positions

Actual selective execution is now a separate runner; see
[REAL_EXECUTION.md](REAL_EXECUTION.md). This document describes the simulator.

This first experiment tests quality and context influence, **not acceleration**.
It uses the existing `confidence_geography.run.collect_sample` decoder and pinned
LLaDA-8B-Instruct weights. It temporarily instruments the model; it neither
trains nor changes weights. It does not use neighborhood/window commitments.

## Conditions

The same saved GSM8K questions run under three conditions:

1. `baseline`: original full recomputation and selection.
2. `delay_only`: fresh states, but sleeping positions cannot commit.
3. `simulated_sleep`: the same scheduling rule, with frozen per-layer states and
   sleeping positions excluded from commitment.

Default policy is TOP1. Threshold is an explicit alternate run, using the
existing rule: commit all awake positions at or above 0.9; if none qualify,
commit the highest-confidence awake position. Ties use the lower position index.
All modes use full-window decoding, never sequential blocks.

Default sleep assignment after a fresh forward:

    still MASK, not committed, full-vocabulary entropy >= 2 nats,
    best non-MASK token probability <= 0.5: skip next 2 forwards

At step t this sets `wake_step=t+3`. Skips do not slide forward while sleeping.
Positions with H<2 remain awake in this initial test. `--low-sleep 1` optionally
adds one-forward sleep for H<1 and p1<=0.5; the intermediate 1<=H<2 band remains
awake. This optional joint rule is disabled by default.

Entropy includes MASK probability. Ranking excludes MASK without probability
renormalization, exactly as in the existing trace collector. Cutoffs are initial
experimental settings, not inferred safety guarantees.

## Where to read the code

- `simulation.py:SleepSchedule`: candidate rule, deadlines, wake and progress.
- `simulation.py:FrozenStates`: attention interception and layer-output hooks.
- `run.py:ProbeModel`: bridges the hooks and unchanged decoder; performs matched
  fresh counterfactual forwards and logs selection differences.
- `run.py:run`: matched inputs, pinned model loading, real-model zero-sleep parity,
  per-condition execution, resume checks.
- `run.py:evaluate`: paired task correctness and aggregate influence statistics.
- `../../confidence_geography/run.py:collect_sample`: the existing decoding loop,
  question-only chat template, unchanged EOS handling and numeric scoring.

## What the hooks actually do

On the first pass every position is fresh. For each block, cache post-RoPE K/V
immediately before SDPA and the block's final residual/FFN output. On subsequent
passes, replace sleepers' K/V before SDPA reads them; restore sleepers' block
outputs before the next block consumes them. Prompt, revealed tokens, and awake
masked positions compute normally. Sleeping input tokens remain MASK.

The input tensor, original positions, attention masks, RoPE, attention backend,
normalization, and weights are unchanged. Sleeping queries/FFN/head still execute
but their results are discarded or excluded from selection. Thus this approximates
the mathematical behavior of a row-selective implementation without implementing
its performance path. A future gathered-row implementation can have different
floating-point rounding and requires its own parity/quality tests.

At wake, all layers recompute. Newly committed positions always compute normally
on the following forward. If every remaining masked position would be asleep,
all are awakened before computation; every such fallback is counted.

The adapter checks the inspected pinned remote source SHA256 and architecture,
and requires one attention/output invocation per block. At process startup it
also compares raw versus instrumented no-sleep logits bit-for-bit on the first
question. A mismatch aborts rather than silently accepting a changed baseline.
This validation adds two model forwards per process, recorded separately.

## Influence diagnostics

By default, each simulated-sleep forward with at least one sleeper also runs a
fresh full-computation control on the **identical current input**. The control
bypasses all cache reads/writes. It is never selected for actual generation.

Per-step diagnostics record:

- Fresh versus frozen top1 token and confidence changes at awake positions.
- Whether the fresh versus frozen awake commitment sets differ, plus Jaccard.
- Which positions/tokens fresh unrestricted selection would commit, including
  sleepers; this measures selection opportunities blocked by the timer.
- Fresh masked-position top1 confidence and entropy for later analysis.

These local comparisons are on the simulated-sleep trajectory, not a separate
baseline trajectory replay. They isolate an immediate stale-state effect at the
same input. Full-run baseline/delay/sleep answers test accumulated effects.
Delay-only and simulated-sleep schedules may diverge after fresh predictions
diverge; their final accuracy difference is not a pure additive causal estimate
of stale context alone.

`--influence-every N` samples diagnostics at step multiples of N with sleepers;
0 disables them. Diagnostic forwards and time are counted separately. The shared
trace's `forward_seconds`/`elapsed_seconds` include diagnostic and hook overhead;
they are not speed metrics. Sleeping position-forwards are opportunities for
future compute reduction, not FLOPs or wall-time savings.

## Scope and potential confounders

- Question text is preserved, wrapped only by the existing model chat template.
  No extra reasoning or answer-format instruction is added.
- Fill all 256 response slots, including special/post-stop slots, as in the
  original decoder. No EOS suppression, early stopping, or future final-answer
  boundary is used to choose sleepers. Final text truncates at the first final
  EOS/EOT; numeric scoring uses a natural #### marker or otherwise the last number.
- Aggregate sleeping/influence counters cover the full response window. Post-stop
  work can inflate apparent opportunity; it must not be presented as answer-only
  savings. Full per-position traces permit retrospective answer-only analysis.
- Existing generic trace metrics can include stale sleeping predictions. Use the
  explicit `sleep` diagnostics, `eligible`, `sleeping`, and `prediction_fresh`
  fields for this experiment, rather than treating all rows as fresh observations.
  Delay-only sleeper predictions are fresh counterfactuals but ineligible.
- Stable uncertainty is not evidence that hidden states are stable. Frozen states
  can distort awake predictions and those differences can accumulate.
- Cache storage, full-tensor copies, extra diagnostic logits, and dense discarded
  computation add memory and runtime. This probe is intentionally not a benchmark
  of a future optimized implementation. Only batch size one is supported.
- Ten questions validate behavior, not a statistically reliable accuracy claim.
  Use a larger held-out matched set before deciding the rule preserves quality.

## Vast commands

From this repository on a BF16 CUDA machine, Python 3.10–3.12:

```bash
RUN_NAME=sleep_smoke_v1 bash scripts/vast_sleep.sh smoke
```

This installs the existing pinned dependencies, runs CPU fixture tests, then
**really loads LLaDA** and runs **2 questions per condition (6 generations)**.
Smoke is not an inference-free test.

Then the 10-question pilot, reusing the installed environment:

```bash
RUN_NAME=sleep_pilot_v1 SKIP_INSTALL=1 bash scripts/vast_sleep.sh pilot
```

One hundred matched questions, only after inspecting the pilot:

```bash
RUN_NAME=sleep_study_v1 SKIP_INSTALL=1 bash scripts/vast_sleep.sh study
```

For threshold decoding, explicitly use another output folder:

```bash
POLICY=threshold RUN_NAME=sleep_threshold_pilot_v1 SKIP_INSTALL=1 \
  bash scripts/vast_sleep.sh pilot
```

To reuse a particular old question set rather than resample GSM8K, add
`SAMPLES_FROM=runs/OLD_RUN/samples.jsonl`; this uses saved order without a reshuffle.
Otherwise seed 1729 selects from a dataset revision pinned in the new manifest.

Direct invocation after activating the installed venv:

```bash
source .venv/bin/activate
python -u -m experiments.token_sleep.run \
  --out runs/sleep_pilot_v1 --samples 10 --policy top1 \
  --length 256 --high-entropy 2 --sleep-confidence 0.5 \
  --high-sleep 2 --low-sleep 0 --influence-every 1
```

Completed samples can resume with the same command; changing arguments or source
requires a new folder. An interrupted sample reruns from its beginning.

Read `runs/sleep_pilot_v1/comparison.json`; inspect individual
`MODE/samples/ID/trace.jsonl.gz` and `result.json` files. The script exports the run
to `exports/` with a checksum. It exports no weights or credentials.

CPU-only verification, no downloads or real model inference:

```bash
python -m unittest discover -s tests -p 'test_token_sleep.py' -v
```
