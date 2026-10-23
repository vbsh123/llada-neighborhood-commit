# 500-question held-out pruning evaluation

Exactly three conditions: uncached/unpruned baseline, progressive audio pruning,
and reference-guided matched-budget generic pruning. No caching, confidence boost,
early/late text substitution controls, drift recording, or fixed-input replay.

Each condition generates once per question with the existing512-slot answer
canvas,32-token blocks,256 maximum forwards and safe EOS stopping. One process-wide
warmup covers all three conditions on the first pending input; it is unscored and
excluded from decode timings. The first two baseline visible answers are also
checked against the full official decoder, outside measured decode time.
The baseline -> audio -> generic order is fixed because the generic control uses
the audio policy's event times/counts but ranks its own remaining prefix. This is
the same control as v21, not an independently tuned generic pruning method.

Pruning settings are frozen: all layers, acoustic start16, semantic start44,
checks every8 forwards, patience2, normalized attention threshold0.5, remove
the lowest-attention25% of qualifying positions at each check. Prompt text is
untouched by the audio policy. Removed positions retain original RoPE coordinates.

Preparation selects500 previously uninspected questions at seed20261024, with
audio durations0–30seconds, excluding all130 IDs frozen in
`inspected_gsm_ids_v22.json` (original30 plus later100). No correctness selection.
Dataset and model revisions remain pinned. On resume the manifest, exclusion
pool, seed/revision/duration provenance and source/config fingerprint are checked.
Completed samples are skipped; an incomplete sample is rerun. A new process
warms once again. Do not pull or edit source during an active job.

Existing Vast checkout with dependencies installed:

```bash
cd /workspace/llada-audio-drift
git pull --ff-only origin audio-drift-study
nohup env SKIP_INSTALL=1 SAMPLES=500 bash scripts/vast_audio_prune500.sh \
  > audio_prune500_v23.log 2>&1 &
tail -f audio_prune500_v23.log
```

On a new machine, clone branch audio-drift-study, then use the same command with
`SKIP_INSTALL=0`; setup/model/data downloads run unattended. Estimated A40 runtime
is roughly8–10hours plus setup, with substantial uncertainty from the new inputs.
To resume, rerun the identical command. Never change counts or settings in an
existing run; use a new RUN_NAME instead.

Outputs: `runs/audio_prune500_v23/`. Every complete sample includes baseline and
policy texts/IDs, original numerical scores, single synchronized decode times,
visible token counts excluding EOS, forward counts, step/commit events, pruning
observations/removals and per-layer work counts. Full hidden states and all
baseline token canvases are not stored. The legacy `median_seconds` field is the
one measured duration, not a median of multiple repeats. Before/after baseline
timings and repeated-policy reproducibility checks are absent in this lean mode.

Reports are generated after completion:

* `analysis/lean_report.md`: original/secondary accuracy, total forwards,
  aggregate TPS, decode speedups, and paired quality changes.
* `analysis/lean_pairs.csv`: one row per condition and question for reanalysis.
* `analysis/numeric_secondary_pairs.csv` and protocol JSON: unchanged frozen
  reference-blind explicit-conclusion-v1 extraction; original scores retained.
* Existing reasoning and pruning reports, plus root input fingerprint.

TPS = summed visible answer tokens / summed synchronized decode seconds.
Pruning's monitoring, selection, gathers and commitment are inside the timer;
model loading, audio preparation, scoring and file writes are outside it.
Single-pass timings are noisier and order is not randomized. TPS, latency and
accuracy should be read together; this is not identical-input forward profiling.
All questions, baseline failures and nonactivated pruning cases stay in the
primary results. A subset with repeated timing can be evaluated separately if
the final differences are close.
