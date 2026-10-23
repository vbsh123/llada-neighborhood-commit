# Development pruning sensitivity

Use the exact50 questions already inspected in v21/v22. This job does not use
the500 unseen heldout_v23 questions and must not be presented as validation.
It reconstructs/reuses the original100-question seed20261023 pool and verifies
the first50 fingerprint against8ed78386500a79daecdf9b9be573efe2c2a4001a5c667b23c58555a6e03205f2.
Changed questions, audio, targets or ordering fail before inference.

Five one-factor configurations, each with an audio and a matched generic policy:

| Threshold relative to uniform attention | Fraction removed from qualifying positions |
|---|---|
|0.5|25% (current reference)|
|0.25|25%|
|1.0|25%|
|0.5|12.5%|
|0.5|50%|

All layers, acoustic start16/semantic44, checks every8 forwards and patience2
remain fixed. Each generic control copies its corresponding audio policy's
event times/counts and ranks its own remaining prefix. It is not an independently
tuned threshold-based generic method. EOS may truncate later planned events.
No caching or confidence boosting.

One baseline plus ten policy generations per question, with safe EOS stopping,
512 answer slots/256 maximum forwards. No per-question warmups or timing repeats.
One process-wide warmup covers baseline and one representative audio/generic
pair. Settings are shuffled deterministically per question, respecting each
audio-before-generic dependency, to reduce systematic setting-order bias.

Only final texts/IDs/scores, synchronized decode seconds, visible token counts,
forward counts, work counts and pruning decisions/activation counts are saved.
Full per-position attention histories and per-forward commitment files are
omitted. No drift, layer/head diagnostics, replay or profiling passes. Attention
observations needed by the policy itself are still computed and included in TPS.
Baseline is measured once per example for a paired reference, rather than copied
from earlier hardware timings. The first two baseline visible outputs are
verified against the official full decoder outside decode timers.

On an existing configured Vast machine:

```bash
cd /workspace/llada-audio-drift
git pull --ff-only origin audio-drift-study
nohup env SKIP_INSTALL=1 bash scripts/vast_audio_prune_sweep.sh \
  > audio_prunesweep50_v24.log 2>&1 &
tail -f audio_prunesweep50_v24.log
```

For a fresh environment use SKIP_INSTALL=0. Setup and input preparation then
run unattended. Expected A40 runtime is approximately2.5–4hours plus setup;
changed output lengths can move it outside that range. Do not run concurrently
with the500-question job on the same GPU. If another job is already running,
finish it before pulling changed source. Identical relaunches resume completed
samples; changing source/settings requires a new RUN_NAME.

Output: runs/audio_prunesweep50_v24. Start with analysis/sensitivity_report.md
and sensitivity_summary.csv: per-setting audio/generic accuracy, paired wins and
losses, aggregate TPS, activation and mean removals. lean_pairs.csv retains
per-example statistics. Frozen secondary scores and original scores are both
saved. Decisions include removed positions and group composition; attention
score arrays are deliberately omitted. Full model hidden states are never saved.

Prefer a consistent tradeoff across nearby settings over the highest development
score. Start times, patience and interval are not tested by this sweep. Freeze
any selected policy before its independent held-out evaluation.
