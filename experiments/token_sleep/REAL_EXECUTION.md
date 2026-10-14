# Real skipping and latency benchmark

This implements actual skipped computation, using the same initial rule as the
simulator: H>=2 nats, best non-MASK p<=0.5, skip the next two forwards. Low-entropy
sleep is disabled. The checkpoint, question-only chat template, full response
window, selection policy and final EOS/EOT truncation match the simulation.

## Implementation

`selective.py:SelectiveForward` gathers awake input rows once. All original LLaDA
blocks then run on that narrower hidden tensor, so their original QKV, residual,
normalization and gated FFN modules process only awake rows. Prompt and committed
tokens are always awake. The input tensor itself is never shortened or renumbered.

Each block's attention method is temporarily replaced. It applies the original
RoPE implementation at the original absolute position indices, scatters fresh
post-RoPE K/V into a per-layer full-sequence cache, and computes rectangular
attention: awake queries against all cached/fresh keys. Original attention methods
are restored even if a forward raises. The same pinned source/architecture guard
as the simulator is required.

No sleeper query, K/V projection, attention row, FFN, or vocabulary projection is
computed. Sleeping layer outputs need not be stored: they never enter the narrowed
layer chain. Waking recomputes all layers from the current input embedding. The
first forward initializes every cache position; an uninitialized sparse call is
rejected. Caches are reset for each question/condition. No gradients are supported.

## Three conditions and fair attribution

1. `original`: ordinary pinned model forward, including its original full-window
   vocabulary projection, and the same compact scoring/selection loop below.
2. `active_baseline`: the new forward with sleep disabled. All transformer rows
   compute, but only currently masked positions receive vocabulary logits.
3. `real_sleep`: the identical new forward with two-forward sleep. Only awake
   masked positions receive vocabulary logits.

The original-versus-real comparison measures the combined practical change.
**Active-baseline-versus-real isolates the additional benefit from sleep.**
Do not attribute the original-versus-active LM-head savings to sleep. Quality and
paired correctness are measured separately for all conditions; gathered matrix
shapes can change floating-point rounding and potentially decoding trajectories.

Only masked positions sleep. Consequently a 54% masked-position sleep fraction
does not imply 54% of transformer rows are removed: prompt and committed rows
remain. The report counts actual active transformer and vocabulary-head rows,
alongside latency. Row reductions are not a FLOP or wall-time guarantee.

## Required GPU validation

Before each benchmark process, `benchmark.py:validate` runs identical
teacher-forced states from the first question through:

- Original versus all-active selective execution.
- Frozen-state simulation versus actual selective execution on the same sleepers.
- An additional sparse-position pattern with changed context, even if early
  uncertainty gating selects no sleepers.

The original decoder supplies validation commitments only, so comparison inputs
remain identical. A teacher-forced committed sleeper wakes on the next state.
Fresh reference calls bypass simulator cache reads/writes. The validation is
excluded from benchmark timing and saved to `validation.json`.

Default maximum full-vocabulary absolute probability difference is 0.02 (two
percentage points), with maximum logit differences and changed token counts also
reported. Both original/all-active and simulator/selective selected commitments
must match on the teacher-forced steps. Failure aborts the benchmark; inspect the
report rather than quietly increasing tolerance. Numerical tolerance is an
explicit initial engineering choice, not a guarantee about downstream accuracy.
CPU fixtures additionally check tighter FP32 agreement, two sleeping forwards,
waking, original positions, grouped-query layouts and actual module row counts.
Full-checkpoint/CUDA validation can only be verified on Vast.

## Timing scope and limitations

- Each condition gets two untimed full-shape warm-up forwards per question, then
  resets its cache. Per-question condition order rotates to reduce systematic
  always-last/warm-GPU effects. This is still an initial measurement, not repeated
  trial confidence intervals or control of other tenants on a shared GPU.
- `decode_seconds` includes the entire full-window decoding loop: scheduling,
  candidate scoring, device transfers, cache updates, selections and commits.
- `model_seconds` measures synchronized model/active-forward sections, including
  gathering/cache work and compact logit extraction, but excluding scoring and
  sleep scheduling. It is not GPU-kernel-only time.
- Model loading, tokenization/input setup, validation, warm-up, serialization and
  console printing are outside those intervals. Cache initialization during the
  first measured forward is included. No diagnostic reference forwards run during
  timed decoding. Traces are collected in memory and written after timing.
- All 256 response slots are filled, including post-stop slots, matching the
  original decoder. Thus these numbers do not measure an early-stopping decoder
  or answer-only latency. No future stop boundary is used for sleeping decisions.
- Thin traces store commitments, timers and work counts, not all vocabulary
  distributions. Use simulator traces for detailed influence analysis.
- This is an initial implementation using PyTorch gather/scatter and ordinary
  SDPA, without custom kernels or CUDA graphs. Irregular shapes/cache traffic can
  erase savings. All-asleep cases explicitly wake all remaining positions and are
  counted. No runtime gain or checkpoint parity is claimed before the GPU run.

## Commands on your existing Vast machine

```bash
cd /workspace/llada-neighborhood-commit
git switch sleep-simulation
git pull --ff-only
source .venv/bin/activate

SAMPLES_FROM=runs/sleep_study_v1/samples.jsonl \
RUN_NAME=real_sleep_smoke_v1 SKIP_INSTALL=1 \
  bash scripts/vast_sleep_benchmark.sh smoke
```

This reuses the first two questions in the saved study, runs GPU parity checks,
then two questions under each of three conditions. It loads the real checkpoint.
If the old sample file is unavailable, omit `SAMPLES_FROM` for deterministic GSM8K
selection. The sample loader records the dataset revision/input hash.

After inspecting validation and the smoke result:

```bash
SAMPLES_FROM=runs/sleep_study_v1/samples.jsonl \
RUN_NAME=real_sleep_study_v1 SKIP_INSTALL=1 \
  bash scripts/vast_sleep_benchmark.sh study

cat runs/real_sleep_study_v1/comparison.json
```

`pilot` selects ten questions; `study` selects one hundred. An explicit threshold
run uses `POLICY=threshold` and a new `RUN_NAME`. Defaults remain TOP1, length 256,
H>=2, p<=0.5, two skipped forwards, seed 1729. No blocks, EOS suppression or
neighborhood commitments are introduced.

Resuming completed samples requires identical source and arguments; changing
either requires a new output folder. An interrupted question/condition reruns
from its beginning. Runs export to `exports/` with checksums.

Read `selective.py` for the model path; `benchmark.py:decode` for scheduling and
timing, `validate` for GPU checks, and `evaluate` for the three comparisons.
