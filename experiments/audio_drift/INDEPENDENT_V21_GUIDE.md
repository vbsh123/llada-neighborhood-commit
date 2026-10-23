# Larger independent evaluation, before combining methods

`scripts/vast_audio_independent.sh` runs **standalone** cache, progressive pruning,
and confidence-priority experiments on the same fixed, new question pool. It
never combines these techniques. Default **50 new questions**, seed20261023;
`SAMPLES` overrides the pool size. Preparation excludes the30 IDs frozen in
`development_gsm_ids_v20.json`, covering the earlier audio development pool.
Old manifests or WAV files need not be copied to a new machine. Dataset revision
is pinned to e94caa2154433a54cd430702666c454977df3083.
Duration selection is unchanged (0–30seconds), with no correctness filtering.

The helper validates exact count, IDs, disjointness, revision, seed, duration
bounds and exclusion provenance whenever a prepared manifest is reused. It does
not overwrite existing inputs. To exclude additional previously inspected pools,
set `EXCLUDE_MANIFESTS` to space-separated manifest paths (Vast paths without
spaces). The original30 are always excluded automatically. A later larger pool
is not automatically new relative to an already evaluated validation pool:
explicitly exclude the additional previously inspected validation manifest.

All methods retain their v20/v16 settings:

* **cache:** shared32,32/8/32, shared50,50/8/50 (A/P/S). Four policies.
* **prune:** progressive audio-only, early-text matched substitution, late-text
  matched substitution, and generic matched removal. Four policies. No caching
  or confidence boost. Threshold.5, patience2, quarter of eligible positions per
  check, interval8, acoustic16/semantic44, all layers.
* **boost:** phased semantic→fixed-prompt odds boost, fixed-semantic and
  fixed-prompt controls at lambda.05/.1, plus zero identity. Seven policies.
  No caching or pruning. Freeze midpoint44/end88 from development; do not
  recompute from eventual validation generation lengths.

Each family has its own paired baseline on the exact same input manifest.
Cache/prune use synchronized generation timings with two repeats after warmup;
replay defaults to **two questions**, not the whole50. Boost is an instrumented
quality experiment, not a clean latency comparison. Every complete sample is
saved, so runs can resume using unchanged config/source/inputs.

## Grading

Original `numeric-labels-comma-v3` scores remain unchanged. After each family,
`numeric_answer_report` runs the frozen **explicit-conclusion-v1** reference-blind
secondary extractor uniformly over baseline and every policy. It accepts common
final labels and unambiguous terminal conclusions, including some answers with
units. It rejects incomplete/ambiguous outputs and does not scan arbitrary
reasoning for the target number. It grades the final expressed value, not full
reasoning consistency; it does not convert units or fix arithmetic.

The secondary rule was developed on the existing v20 outputs, before new
validation results. Both primary and secondary tables must be reviewed; marker
failures should not be presented as mathematical reasoning failures. Preserve
the extractor version/source hash in `analysis/numeric_secondary_protocol.json`.
CSV stores extraction type, predictions and original scores. Grader limitations
remain; ambiguous cases require a uniform audit across all methods.

## Commands

Existing Vast checkout/environment:

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
SAMPLES=50 bash scripts/vast_audio_independent.sh prepare
nohup env SAMPLES=50 REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_independent.sh cache > independent_cache50_v21.log 2>&1 &
```

Use `prune` or `boost` for separate jobs after the prior job completes. `all`
runs cache, then prune, then boost sequentially, with no simultaneous GPU jobs.
Outputs: `runs/audio_independent_cache50_v21`, `..._prune50_v21`,
`..._boost50_v21`. Change `RUN_PREFIX` if changing an experiment configuration.

## Two-machine split

Prepare on each machine with the same code, `SAMPLES` and seed. Compare the
printed **INPUT_FINGERPRINT** values: they cover ordered question IDs, pinned
revision, exact WAV bytes, prompts and targets, excluding machine-specific paths.
They must match. Each completed run includes `input_fingerprint.json`.

Machine A: `SAMPLES=50 bash scripts/vast_audio_independent.sh prune`.
Machine B: `SAMPLES=50 bash scripts/vast_audio_independent.sh cache-boost`.
The latter runs cache then boost sequentially, never together. Prefer the same
GPU model and software on both. Compare timings against each family's own
baseline; absolute seconds and cross-family speed rankings across different
hardware are not controlled comparisons.

The larger suite is substantial: the measured20-question v20 pruning and boost
runs took about105 and61minutes. With their full policy sets,50 questions would
scale to roughly4.4 and2.6hours respectively, **plus the separate cache run**;
actual lengths/activation affect this. Run families independently if desired.
This helper does not claim a2–3hour budget for all three larger evaluations.

Validation is independent only until results guide further choices. After
selecting winners, subsequent tuning requires another new evaluation cohort.
These runs choose standalone candidates before any combined implementation.
