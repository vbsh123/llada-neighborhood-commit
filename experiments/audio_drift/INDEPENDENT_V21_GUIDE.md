# Larger independent evaluation, before combining methods

`scripts/vast_audio_independent.sh` runs **standalone** progressive pruning
and confidence-priority experiments on the same fixed, new question pool. It
never combines these techniques. Default **100 new questions**, seed20261023;
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

* **prune:** progressive audio-only, early-text matched substitution, late-text
  matched substitution, and generic matched removal. Four policies. No caching
  or confidence boost. Threshold.5, patience2, quarter of eligible positions per
  check, interval8, acoustic16/semantic44, all layers.
* **boost:** phased semantic→fixed-prompt odds boost, fixed-semantic and
  fixed-prompt controls at lambda.05/.1, plus zero identity. Seven policies.
  No caching or pruning. Freeze midpoint44/end88 from development; do not
  recompute from eventual validation generation lengths.

Each family has its own paired baseline on the exact same input manifest.
Pruning uses synchronized generation timings with two repeats after warmup;
replay defaults to **two questions**, not the whole100. Boost is an instrumented
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

## Two-machine unattended commands

On each new machine, clone branch `audio-drift-study`; on an existing checkout,
pull the latest commit. The outer `nohup` covers setup, dataset preparation,
model downloads, generation and reports, so no later manual launch is required.

Machine A:

```bash
nohup env SAMPLES=100 bash scripts/vast_audio_unattended.sh prune \
  > independent_prune100_v21.log 2>&1 &
```

Machine B:

```bash
nohup env SAMPLES=100 bash scripts/vast_audio_unattended.sh boost \
  > independent_boost100_v21.log 2>&1 &
```

Machine A runs four standalone pruning variants and paired baseline. Machine B
runs seven confidence-priority variants and baseline. **Neither launches any
caching experiment.** `vast_audio_independent.sh all` now means pruning then
boost only. The optional explicit `cache` mode remains available for future use;
the old `cache-boost` queue name is removed to avoid launching the canceled job.

Setup installs/verifies dependencies and the pinned DIFFA checkout, then
preparation creates the100-question manifest automatically. It excludes the
bundled30 development IDs without requiring old audio folders. On existing
complete environments, `vast_audio_independent.sh prune` or `boost` skips setup.
Both launchers default to100; explicitly setting SAMPLES is recommended.

Compare the printed **INPUT_FINGERPRINT** values: they cover ordered question
IDs, pinned revision, exact WAV bytes, prompts and targets, excluding
machine-specific paths. They must match. Each completed run includes
`input_fingerprint.json`. Prefer the same GPU model/software on both. Compare
latency against each family's own baseline; absolute seconds across different
hardware are not controlled comparisons.

Outputs: `runs/audio_independent_prune100_v21` and
`runs/audio_independent_boost100_v21`. All reports preserve original scores and
add the frozen reference-blind secondary numerical evaluation. No techniques
are combined. Use a new RUN_PREFIX if changing experiment configuration.

Measured20-question v20 pruning and boost runs took about105 and61minutes.
With full policy sets,100 questions scale to roughly **8.7 and5.1hours**
respectively on the same hardware, plus setup/downloads. Actual generation
lengths and activation change these estimates. Running on two machines reduces
elapsed time but does not make either individual job shorter.

Validation is independent only until results guide further choices. After
selecting winners, subsequent tuning requires another new evaluation cohort.
