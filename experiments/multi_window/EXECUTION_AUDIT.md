# Execution audit after the first 100-question region-window study

## What is established

The matched reports have 25,600 forwards each. Vanilla took 1,454.488 s of decode
and 1,280.931 s inside synchronized model calls; multi-window took 1,401.072 s and
1,310.455 s respectively. That is 3.8% higher throughput overall, but model-call
execution itself was 2.3% longer. The model-call timer includes input gathering,
cache maintenance, vocabulary-head computation and host dispatch, not GPU kernels
alone. Serialization is outside the decode timer. Do not attribute every
non-model second to scoring: it also includes scheduling, transfers and logging.

The scheduler has the intended four-forward refresh, gap=4 region separation,
all historical regions awake, no filled response rows asleep, and unioned padded
windows. Existing toy-model tests compare sparse cache mathematics against the
dense frozen-state simulator. No semantic error explaining the five-point
accuracy drop has been established. Strict parity on the short GPU validation
prefix is not full-trajectory equivalence to vanilla or a quality guarantee.

## Confirmed avoidable execution work, now removed

In `experiments/token_sleep/selective.py` before this audit:

1. Every full-refresh layer gathered sine and cosine tables by identity indices.
   With 32 layers that is 64 unnecessary positional gathers per full refresh.
   Full forwards now use native full-sequence RoPE instead.
2. Sparse forwards repeated identical positional-table gathers in every layer even
   when pinned LLaDA's layers share those buffers. Tables are now gathered once
   per distinct source-buffer geometry/dtype per forward. Distinct buffers are
   never conflated, and references prevent allocator pointer-reuse aliasing.
3. Prediction-membership checks searched a Python list for each prediction,
   creating O(active_rows * predictions) work. Validation now uses set membership.
4. CPU-known position indices were transferred to the GPU and remapped with
   `searchsorted`. Mapping is now constructed on CPU and transferred directly.
   Fully awake forwards also avoid gathering the complete input by identity.

These are implementation inefficiencies, not proof of the cause or magnitude of
GPU slowdown. QKV weights, attention context, cached-state semantics, scheduling,
commitment policy, prompts and generation dtype are unchanged. CPU FP32/BF16
old-versus-new tests check exact logits and cached K/V across sparse patterns and
full wakes. CPU operator counts verify removal of redundant gathers. They do not
predict a GPU speedup or establish full checkpoint numerical equivalence.

The original checkpoint source was inspected at revision
`08b83a6feb34df1a6011b80c3c00c7563e963b07`:
https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/08b83a6feb34df1a6011b80c3c00c7563e963b07/modeling_llada.py

## Short Vast diagnostic, not another quality study

This reads a single saved question's trace, reconstructs its original states and
sleep geometry, and replays the first 32 forwards three times. It compares:

- `original`: normal full-sequence `model(x)`, retaining logits for the same awake
  positions as the selective paths (as in our vanilla benchmark).
- `legacy_selective`: frozen selective source from commit `377223c`.
- `updated_selective`: current selective execution with the changes above.

Old and new selective paths have independent caches initialized from identical
inputs. Their probabilities/predictions are compared after timing. The vanilla
path uses fresh global context, so its outputs are not expected to match the
stale-K/V paths; its masked prediction candidates are matched for timing only.
No decoding decisions are generated; original saved commits teacher-force input
states. No original run files are changed.

```bash
cd /workspace/llada-neighborhood-commit
git switch sleep-simulation
git pull --ff-only
source .venv/bin/activate

python -m experiments.multi_window.profile \
  --run runs/multi_window_study_v3 \
  --out runs/multi_window_profile_v1 \
  --steps 32 --repeats 3 --capture
```

No package installation should be needed on the machine that completed the study.
The old source is loaded from local Git history, not downloaded. A shallow checkout
must contain commit `377223c`; otherwise fetch repository history first. The output
folder must be new. `--sample-id 00654` optionally selects a specific question;
otherwise the first saved sample is used. The checkpoint may still download if
its cache is absent. No dataset download is needed.

Output:

- `profile.json`: same-input model-call wall/CUDA-event timings, divided into
  all-awake and sparse geometry, old/new probability error, token winner changes,
  GPU, source hash and individual calls.
- `operators.txt`: CPU/CUDA operation summary.
- `operators.json`: Chrome-compatible operator trace with shapes and mode scopes.

`--capture` makes a separate untimed operator capture on the first sparse saved
forward, with cache initialization replayed first. It does not contaminate the
reported wall/CUDA timings. If CUDA profiler support is unavailable, omit
`--capture` and choose another new output folder.

This diagnostic is one-question execution profiling, not 100-question runtime or
quality evaluation. If old/new probabilities or winners differ, investigate
before treating performance as a purely execution-preserving change. No numerical
failures are silently waived. New generation studies must use fresh output names
because source fingerprints intentionally change.
