# Generation-dependent attention crossing pilot

Run `bash scripts/vast_audio_drift.sh gsm-crossing-prune` after setup. Default:
two clips >=15 seconds from the existing 30-question manifest, no correctness
filter. This is a development pilot, not fresh held-out evaluation.

Two policies plus uncached baseline:

* **Audio:** check layer 25 every eight forwards, starting at index 8. Each
  group's summed attention to currently surviving acoustic/semantic positions
  is compared independently with summed attention to prompt-text positions.
  Two consecutive checks below prompt trigger the next pruning tier. Remove
  lowest-attended positions to reach cumulative 25%, 50%, then 75% of the
  original group. Reset that group's patience after each event. No count cap.
  The first possible event is index 16; maximum default removal takes six
  qualifying checks (index 48). Events persist through layers 25–31 and later
  forwards. Positions survive/recompute normally in layers 0–24 each forward.
* **Full-prefix control:** copy the audio run's event times and number removed
  per event; choose its own lowest-attended positions over acoustic + semantic
  + prompt text. This tests selection scope at equal scheduled removal budgets.
  It is a reference-guided control, not an independently deployable trigger.
  If it stops earlier its autonomous workload need not match; equal-input
  replay asserts matching computed-row/QK budgets. Replay derives its own pair
  of schedules from the paired replay source, not autonomous generation.

Scores average all heads and 64 evenly spaced fixed answer-canvas queries,
including masks and future blocks; no retrospective final-answer probe positions.
Attention is sampled at the first pruned layer, not averaged across all seven.
This is total group mass, not mass per token. Counts differ. Pruning itself can
reinforce the crossing by reducing group size. Independent acoustic/semantic
checks run simultaneously from the same pre-event attention snapshot. No
reference-answer, final length or correctness information controls selection.

All retained rows refresh every forward. Rows and K/V columns are physically
compacted between blocks, using original RoPE coordinates. No temporal caches.
Final norm/head keep the full output canvas, as in baseline. Only seven layers
are affected by default; large total speedups are not guaranteed.

Every policy includes selection/monitoring overhead in latency. Control runs
repeat scoring at the source's logged check indices, even without a drop.
Audio source must precede control; other order randomization is preserved.
Warmups + two synchronized timings each for generation and equal-input replay.
All timed repeats must reproduce positions/checks/output tokens. Reports retain
non-triggered cases and baseline failures. Generic trigger masses are logged but
do not decide control events. Read `cache_benchmark.json`, `fixed_input_replay.json`,
`analysis/pruning_pairs.csv`, and their `pruning.trigger_checks` / `selections`.

Settings: `CROSSING_FRACTIONS="0.25 0.5 0.75"`, `CROSSING_LAYER=25`,
`CROSSING_START=8`, `CROSSING_INTERVAL=8`, `CROSSING_PATIENCE=2`,
`PRUNE_QUERIES=64`, `SAMPLES=2`. `MANIFEST` bypasses duration selection.
Use a new run name after changing configuration. Operator trace mode for the
older staged experiment is deliberately unavailable here; clean timing and row/
QK accounting remain enabled. CUDA performance must be measured on Vast.
