# Compact staged pruning (v17)

This is a pruning-only pilot: retained representations recompute every forward.
It replaces the exploratory v16 implementation's full KV/hidden banking with
physically compact hidden rows between transformer blocks. No model weights or
upstream source files are changed. It is not yet a measured CUDA speedup.

## Policies and controls

| Pattern | Acoustic | Semantic | Fixed prompt text |
|---|---|---|---|
| A | Prune from layer0 after16 forwards | Retain | Retain |
| AS | Same | Prune from layer16 after32 forwards | Retain |
| ASP | Same | Same | Prune from layer16 after32 forwards |

Defaults use two removal fractions:50% and75% of EACH selected group's original
positions. With64 acoustic,210 semantic and76 text positions,50% means32,
105 and38 positions respectively. There is no shared63-position cap. Fractions
up to100% are supported. All answer slots remain. Selecting a fraction ranks
positions by current mean attention; these are not absolute attention cutoffs.

Every pattern/fraction has a generic full-prefix counterpart. At each layer/time
boundary it removes the SAME count as the group-aware counterpart, but ranks all
remaining prefix positions, including audio and text. Same-boundary selections
use a common score snapshot with disjoint selected positions. Generic and group
replays assert equal row/query-key budgets at identical input forward counts.
Total:3 patterns x2 budgets x2 selectors=12 policies, plus an uncached baseline.
No claim of group specificity is justified unless the group-aware patterns beat
these controls at a useful quality/latency tradeoff.

Acoustic removal persists through all layers. Semantic/text positions removed
at layer16 still participate normally in layers0–15 on every subsequent forward.
They are recomputed from input embeddings again in those earlier layers, rather
than frozen or revived from stale hidden states. Deeper layers omit them.

## Execution and costs

Before activation, unaffected blocks call upstream forward directly, with no
bank copies. At a selection boundary, QKV is computed before selecting tokens;
attention queries/KV and MLP are compacted immediately. On later forwards,
existing removals are applied before QKV at the corresponding layer boundary.
Every surviving row is fresh. Original absolute RoPE coordinates are preserved.
Host position maps are cached; no full position tensor is copied back to CPU
per layer/forward. Attention selection uses all heads and16 fixed answer-canvas
probes including masks. It never uses the final answer span or future trajectory.

The final block restores the original output shape ONCE per forward for the
unchanged final norm/vocabulary head and answer-logit indexing. Removed prefix
outputs are zero-filled and never influence answer logits through the row-wise
head. Embeddings/final norm/vocabulary work still processes the full input shape;
this is not a vocabulary-head optimization. Selection and output restoration
costs remain inside clean timing. QKV counts at selection boundaries are recorded
separately from attention/MLP counts. Tiny upstream FP32/BF16 tests compare answer
logits with an independently masked dense reference, including full group removal.
Full-size CUDA quality/speed still need the pilot.

## Small pilot first

`gsm-staged-prune` defaults to2 examples: the first rows of the existing pinned
30-example manifest with audio duration>=15 seconds. Selection uses duration,
not correctness. The current manifest chooses01156 and00408. Audio paths are
resolved and provenance retained. This is development data, not held-out validation.
Set an explicit MANIFEST to use other inputs; an explicit manifest bypasses the
pilot duration filter. Full-size CUDA execution has not been locally tested.

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_gsm_staged2_v17 SAMPLES=2 SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  WHISPER_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1 \
  bash scripts/vast_audio_drift.sh gsm-staged-prune \
  > audio_gsm_staged2_v17.log 2>&1 &
tail -f audio_gsm_staged2_v17.log
```

512 answer slots,256 maximum forwards,32-position blocks, fixed-quota commitments,
safe EOS stopping, a warmup plus two timed autonomous runs per policy and two
identical-baseline-input replay repeats. Forward16/32 correspond to32/64 revealed
canvas slots (6.25%/12.5%), not percentages of the unknown final answer length.
Each selected stage records actual unmasking metadata from that policy's decode.
Policy and replay trajectories may select different positions; replay is not graded.

Set `STAGED_FRACTIONS="0.5 0.75"`, `ACOUSTIC_START=16`, `LATER_START=32`,
`DEEP_LAYER=16`, `PRUNE_QUERIES=16`, `SAMPLES`, `REPLAY_STEPS` etc. explicitly
before a new run name to change the pilot. Later start must be >= acoustic start;
deep layer must be>0. Semantic/text are not refreshed on a cache schedule.

## Operator attribution

The first pilot example gets separate operator captures for baseline and all six
patterns/controls at the smaller fraction, at initialization, selection boundaries
and the immediately following forward. This replays input history with fresh
contexts and adds attribution outside clean latency measurements. Use
`PRUNING_PROFILE_SAMPLES=0` to omit it.

`pruning_traces/index.json` and Chrome traces/operator tables include QKV, MLP,
attention/output projection, vocabulary, `prune.selection`, `prune.compact_rows`
and `prune.restore_output`. CPU annotation aggregates are retained once to avoid
duplicate CUDA annotation spans. Scopes and their child kernels must not be added
together. Norms/activations/rotary/dispatch also contribute. Instrumented timings
are attribution, not clean throughput. CUDA tracing still requires GPU validation;
CPU tests validate the capture/restoration and expected scope labels.

Download the entire run plus console log. Benchmark/replay JSONs retain full
policy/stage specs, answers/scores/repeats/actual row and query-key counts. The
pruning report sums removal counts and records per-stage details; it does not sum
attention masses across different layers. Stage activation is visible when EOS
prevents a later stage from being reached. Primary scores retain all examples.

## Attention interpretation

A full semantic group receiving12–21% of all attention is different from its
lowest-attention32 positions receiving~0.31% at one layer/forward. Those numbers
are baseline sampled-query means, not a probability that the group is important
or a pruning guarantee. Current numerical scoring still needs a uniform format
failure audit; keep strict scores and generated text to support later rescoring.
