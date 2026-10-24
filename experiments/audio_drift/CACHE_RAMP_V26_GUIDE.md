# Scheduled cache reuse: semantic versus fixed prompt text

This is a development-only cache experiment on the exact50 previously inspected
questions. It is separate from the500-question pruning jobs. No pruning,
confidence boosting, attention observation, drift, replay or operator profiling.
All answer rows remain fresh on every forward; all KV columns remain present.

Acoustic refresh stays at16. Compare these seven conditions, plus a clean
baseline, with one timed generation per condition/question:

| Policy | Acoustic | Fixed prompt | Semantic |
|---|---|---|---|
| Shared starting control |16|8|8|
| Semantic ramp |16|8|8 ->16 ->32|
| Prompt ramp |16|8 ->16 ->32|8|
| Constant semantic16 |16|8|16|
| Constant semantic32 |16|8|32|
| Constant prompt16 |16|16|8|
| Constant prompt32 |16|32|8|

The common8 control is shared between the two target groups, not rerun twice.
The prompt ramp applies to fixed prefix text, not generated answer text.
Schedules match across groups, but their token counts differ: this is not a
compute-budget-matched modality comparison. Actual row counts and group refresh
counts are saved to make that difference visible.

Phase starts are frozen at zero-based forward indices0,48,96. Development v22
baseline used4688 forwards on50 examples (mean93.76; half46.88). Half-mean and
mean were rounded upward to multiples of8. No current sample's future EOS or
final answer length is used. At each phase entry the target group refreshes;
subsequent refreshes use8,16,32 steps since that phase start. With these aligned
boundaries the refresh steps are0,8,...40;48,64,80;96,128,160,... .
This is scheduled progress-dependent reuse, not an online attention/drift detector.

The existing compact cache path reuses prefix K/V and post-block hidden states,
skipping their QKV, query/output and MLP row updates on nonrefresh forwards.
Original RoPE positions and full vocabulary head are preserved. No extra
inference-time probes are needed to decide the frozen schedule.

Minimal outputs: final texts/IDs, original and frozen secondary scores,
synchronized decode time, visible output token counts excluding EOS, forward
counts, group refresh counts and per-layer work counts. No per-step traces or
hidden-state snapshots. One process-wide baseline/cache warmup; first two
baseline visible outputs also checked against official generation outside timing.
Policy order is shuffled deterministically by question. No per-example timing
repeats. Audio preparation, grading and file I/O stay outside decode timers.

Do not pull changes into a checkout used by an active pruning process. Use a
separate checkout or another machine:

```bash
cd /workspace
git clone --branch audio-drift-study \
  https://github.com/vbsh123/llada-neighborhood-commit.git llada-cache-ramp
cd llada-cache-ramp
nohup env SKIP_INSTALL=0 bash scripts/vast_audio_cache_ramp.sh \
  > audio_cacheramp50_v26.log 2>&1 &
tail -f audio_cacheramp50_v26.log
```

Use SKIP_INSTALL=1 for a prepared checkout, or set AUDIO_VENV to an existing
compatible environment. Estimated A40 runtime is roughly1.5–2.5hours plus setup
and downloads; answer length changes can increase it. Do not run alongside the
pruning jobs on the same GPU.

Output: runs/audio_cacheramp50_v26. Read analysis/lean_report.md for accuracy,
TPS, paired losses/recoveries and latency. analysis/cache_ramp_report.md and
cache_ramp_pairs.csv show phase entry counts and actual refresh counts. Early
EOS examples remain in primary scores even if they never reach a ramp boundary.
The report checks that ramps reproduce the shared8 control when generation ends
before the first transition. cache_ramp_provenance.json records boundary choices.

Compare each ramp with both its constant16/32 controls and the shared starting
control; otherwise improved speed could merely reflect more caching. This is
development data; freeze candidates before any later held-out validation.
