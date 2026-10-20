# Diagnosing audio-cache speed headroom

The `gsm-diagnose` mode separates quality, continuous replay, per-forward refresh
cost, operator attribution, and representation/attention diagnostics. It does not
change DIFFA weights or the compact cache implementation.

## Why interval 8 already skips most audio updates

The cache initializes at forward index 0. Interval r refreshes at indices
0, r, 2r, ... . For F>0 forwards, refresh count is 1+floor((F-1)/r).
Permanent caching (r=0) initializes once. For F=64:

| Interval | Refresh forwards | Reused forwards | Audio-row updates skipped |
|---|---:|---:|---:|
| 8 | 8 | 56 | 87.5% |
| 16 | 4 | 60 | 93.75% |
| 32 | 2 | 62 | 96.875% |
| Permanent | 1 | 63 | 98.4375% |

Moving 8 to 16 halves the remaining audio refresh work, not the total workload.
All 512 answer rows still compute on every forward, alongside the full vocabulary
head and complete KV columns. The 30-question run has 76 prompt positions,
64 acoustic positions, and 126-312 semantic positions. Shared caching can already
capture most available prefix-row savings.

`budget.py` counts exact token-layer updates including initialization and finite
forward counts. Its linear-projection, QK/AV, and vocabulary FLOP model omits other
operations and hardware effects. Ratios from that model are NOT latency bounds.
The permanent-audio condition measures practical headroom in this implementation;
it is not an architecture-independent theoretical upper bound.

## Controls

Seven conditions plus the uncached baseline:

- refresh-every-forward identity;
- shared prefix every 8 forwards;
- A8/P8/S16 (isolate semantic interval change);
- A16/P8/S8 (isolate acoustic interval change);
- A16/P8/S16;
- A32/P8/S32;
- A0/P8/S0 (audio initialized once, prompt refreshed every 8 forwards).

No depth variants. All use the same 512 answer positions, 256 maximum forwards,
32-position decoding blocks, seed, input examples, and EOS stopping rule. The
commitment rule is fixed-quota confidence ranking (two positions per forward),
not variable threshold commitment. Autonomous outputs can have different lengths
and forward counts. Baseline failures remain in quality reports.

## Run on Vast

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_gsm_diagnose6_v11 SAMPLES=6 SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh gsm-diagnose \
  > audio_gsm_diagnose6_v11.log 2>&1 &
tail -f audio_gsm_diagnose6_v11.log
```

Uses the first six interleaved examples from the existing 30-question manifest
(three development, three fresh); it creates that manifest if absent. These are
now exploratory examples, not held-out validation. Inspect the first completed
sample before leaving the run unattended.

Defaults: two timed autonomous decodes per condition after warmup, two continuous
replay repeats (up to 128 identical pre-EOS states) per condition on all six
examples; separate synchronized per-forward timings with three repeats on the
first two examples. `REFRESH_PROFILE_SAMPLES`, `REFRESH_PROFILE_REPEATS`,
`REPLAY_STEPS`, `DIAGNOSTIC_SAMPLES`, and `DIAGNOSTIC_QUERIES` are adjustable.
Set `REFRESH_TRACE=0` to omit instrumented operator captures. For a one-question
smoke check use a NEW run name and `SAMPLES=1`; refresh profiling automatically
caps its default sample count at one. Full 32-layer attention diagnostics are
collected by default. `DIAGNOSTIC_LAYERS=7,15,23,31` is an explicit smaller option.

## Outputs and interpretation

- `cache_benchmark.json`: autonomous answer text, scores, forward counts and
  warmed synchronized decode timings, without diagnostic hooks.
- `fixed_input_replay.json`: continuous matched-state/forward replay. This is the
  primary comparison for forward-path speed, independent of output length.
- `refresh_profile.json`: separate per-forward synchronized wall/CUDA timings,
  initialization/cache-hit/prompt-only/full refresh phases, actual row counts and
  estimated FLOPs. Synchronizing every forward changes dispatch behavior; do not
  substitute these times for continuous replay or autonomous generation.
- `refresh_traces/`: separate representative forwards at indices 0,1,8,16,32 when
  those states exist. Cache history is built outside the capture for intervening
  forwards. Baseline, shared8, 16/8/16, and 32/8/32 are captured. The index labels
  QKV, feed-forward projections, attention kernel, and vocabulary scopes; operator
  tables also expose copies, gathers, norms and nonlinear operations. Captured
  scope/CPU durations are NOT speed measurements. Do not add parent scopes and
  their child operators, or double-count enclosing CUDA annotation spans.
- `attention_dynamics.jsonl`: separate untimed uncached baseline replay through
  safe EOS. At each measured layer, FP32 softmax reconstructs attention from four
  fixed answer-query positions to all keys. No full attention matrix is retained.
  Positions are sampled across the baseline visible answer span retrospectively;
  selection never changes model inputs or predictions. Only queries with the
  same input identity on both consecutive forwards enter comparisons. Query
  representations can still change: query cosine drift is reported separately.
- `attention_token_metrics/`: compressed scalar per-position hidden cosine drift,
  attention weight and absolute attention change. No hidden-vector archive.
- `analysis/`: accuracy tables, token/forward/refresh budgets, measured phase
  timings, drift-progress/attention CSVs, correlations, and plots.

Attention metrics include raw group mass and L1 change, within-group conditional
Jensen-Shannon divergence (only non-negligible masses), and group-specific value
contribution change before the attention output projection. Attention weights can
stay fixed while values change; do not equate weight stability with irrelevance.
The hidden vector is the block output after attention and MLP. Correlations across
positions and across time are distinct outputs. Time correlations are stratified
by example, layer, and group, avoiding pooled depth effects. They are observational,
not causal evidence that a particular drift component changes answer quality.

Global canvas revealed fraction is coupled to forward index under fixed quotas.
Current block revealed fraction describes a different phase variable. A third
axis measures the revealed fraction of the final pre-EOS answer retrospectively;
it uses the final answer span only for analysis, never for decisions or input.
Different progress bins can contain different examples; inspect the per-sample
bin CSV before attributing a late-bin change to progress. Plots average
within each sample/depth-group/progress bin, then average samples equally; they
exclude age-zero comparisons and post-content-completion tails. They do not
establish that revealing tokens causes drift.

## Analyze existing data without generation

```bash
python -m experiments.audio_drift.diagnostic_report --run runs/audio_gsm_night30_v10
python -m experiments.audio_drift.diagnostic_report \
  --run exports/audio_gsm_depth10_v9.zip --out analysis/gsm_depth10_v9_progress
```

ZIP input is streamed directly. New attention observations require a model run;
old hidden/KV drift files cannot recover attention distributions without queries.
Completed policy results are written incrementally. An interrupted example remains
in a hidden `.incomplete` directory; completed examples are reusable with identical
source/settings fingerprints. Download the whole run directory, including hidden
files for partial runs, plus the console log.

## VidLaDA comparison

MARS-Cache combines modality/depth refreshing with frame-local attention and
global anchor tokens. Its headline 8-12x acceleration and 1.3-1.5x throughput over
Dual-Cache are combined-system comparisons, not pure modality-refresh gains.
Table 9 separately compares shared modality intervals 32/32/16/8 with doubled
vision intervals: Ego throughput 12.0 to 16.4 TPS, MLVU 12.2 to 16.7 TPS, with
slightly different accuracies. That ablation supports modality refreshing, but is
not an equal-input replay or a result on DIFFA. A much larger video prefix can
leave more refresh work after shared caching; remaining compute proportions,
answer-token treatment, backend and timing protocols all matter.

Source: https://arxiv.org/html/2601.17868v2 (sections 4.2, 5.2, F.5).

## Local validation

CPU tests use the actual pinned tiny upstream transformer: attention diagnostics
preserve logits, changing query identities are excluded, compact row counts match
the analytical budget, and profiler scope restoration preserves cached outputs.
Directory and ZIP analysis agree on a synthetic replay fixture. A shell fixture
checks the launcher through the real runner argument parser. Full-size CUDA
profiling and BF16 execution still require the first Vast run.

## Does attention transition from low to high, or high to low?

The updated recorder (attention schema 2) saves mean and maximum attention over
all heads and the SAME fixed answer-probe positions on every forward, including
forward zero and steps where probe token identities change. The pairwise drift
correlation still uses only unchanged query identities; those conditional means
must not be used as a fixed-population time series. Older schema-1 runs are marked
unsupported for recurrence rather than silently mixing query populations.

`gsm-diagnose` now automatically runs `attention_recurrence`. It tracks acoustic
and semantic token positions separately, per layer and per example. Defaults:

- Low: mean attention <= 0.5 times uniform attention (uniform = 1/Nkeys).
- Future high: attention >= 2 times uniform, measured both as a mean and as a
  maximum over any measured head/query to expose spikes hidden by averaging.
- Require 1, 4, or 8 consecutive low-attention forwards. Trigger once per
  contiguous low episode for a token, at the first eligible point that is not
  already high under the chosen high metric.
- Measure return to high attention within 4, 8, 16, or 32 subsequent forwards,
  plus any time during the observed remaining decode. Fixed horizons require
  complete future observations; insufficient remaining steps are right-censored,
  not treated as evidence of safety.

Both directions are reported. High-to-low requires consecutive high forwards
and measures the first subsequent low observation. For the peak-based version,
the peak itself must fall below the low threshold. A first low observation is
not evidence that attention stays low; inspect low-to-high recurrence separately.

Outputs: `attention_recurrence_per_sample.csv` (direction, numerators, denominators, censored
counts and first-target delay histograms), `attention_recurrence_by_layer.csv`,
`attention_recurrence_summary.json`, a notes report, and a layer-wise probability
plot for low-to-high plus `attention_high_to_low.png`. Results include pooled episode probabilities and equal-sample means, plus
stratification by the anchor's quarter of baseline decoding. Repeated episodes
and layers are dependent; episode counts are not independent sample counts.

The thresholds are descriptive starting points, not calibrated pruning rules.
They can be changed after collection without additional model forwards:

```bash
python -m experiments.audio_drift.attention_recurrence \
  --run runs/audio_gsm_diagnose6_v12 \
  --out runs/audio_gsm_diagnose6_v12/analysis/recurrence_low025_high4 \
  --low 0.25 --high 4 --histories 1 4 8 --horizons 4 8 16 32
```

Only sampled answer queries are covered. Low weights here do not prove that all
answer queries ignore the token, or that token removal preserves predictions.
The attention distribution can change after removal. This is a baseline risk
measurement that should precede a causal pruning intervention. Use the updated
recorder in a new run (for example `RUN_NAME=audio_gsm_diagnose6_v12`); avoid
changing source files during an active run.


## Retention inventory: what can be analyzed after the GPU run?

Defaults below refer specifically to `gsm-diagnose`, not older experiment modes.
All six examples receive quality benchmarks, matched-input replay and separate
attention diagnostics; only the first two receive synchronized refresh profiling
and representative operator traces. Attention diagnostics cover all 32 layers
and baseline states through safe EOS. Continuous speed replay uses at most the
first 128 pre-EOS baseline states. Attention collection is untimed and separate
from clean speed measurements.

| Saved data | Granularity | Further CPU analyses |
|---|---|---|
| Input IDs, source/revisions, audio hashes, prompt, target, cohort, duration, group counts/positions and run settings | Example/run | Length/cohort stratification; reproducibility checks |
| Baseline `steps.json`: block, masks, committed positions, commitment counts | Every full-baseline forward | Actual forward counts, block phases, unmasking curves, baseline token commit order |
| Final baseline/policy text, token IDs, strict/normalized scores, parse failures and EOS flags | Example/policy/repeat | Alternative answer parsing, paired correctness, output lengths and tokens/second |
| Clean decode and matched-state replay times, peak allocated memory, row accounting | Example/policy/repeat | Quality/latency comparisons, repeat variability, exact refresh counts |
| Synchronized wall/CUDA forward timings and refresh phases | First two examples, policy/forward/repeat | Initialization versus cache-hit/refresh costs |
| Representative CPU/CUDA operator traces with shapes and labeled scopes | Selected states on first two examples | Kernel/MLP/projection/attention/vocabulary attribution; avoid nested-scope double counting |
| Hidden cosine distance and relative L2; key/value cosine distance | Example/layer/prefix group/consecutive-forward pair | Drift versus progress, depth, task/example characteristics |
| Attention mass, mass change, raw L1 change, conditional JS divergence, value-contribution cosine/L2 change, query cosine drift | Same, for unchanged-input probe queries | Weight versus value changes; drift/attention/progress correlations |
| Original prefix positions and fixed-probe mean/peak attention | Every prefix token, layer and observed forward, including zero | New low/high thresholds, histories/horizons, token rankings and recurrence analyses |
| Per-token hidden cosine distance, attention absolute change and conditional mean | Consecutive forwards with unchanged-input probe queries | Within-group token drift/attention correlations and candidate ranking |

The prefix groups are acoustic audio, semantic audio and fixed prompt text.
Answer-row hidden drift is not collected here. Fixed-probe scalar arrays include
prompt positions, although the current recurrence report selects acoustic and
semantic positions only; prompt recurrence can be added offline from the same
arrays. Per-token relative L2 and per-token key/value drift are not retained in
this mode; their group summaries are available as described above.

The fixed-probe arrays average over all measured queries and heads and also save
the maximum across that population. They do NOT retain individual head/query
weights, full hidden/Q/K/V vectors, logits, or full attention matrices. Therefore
head-specific analyses, unsampled-query coverage, hidden-vector PCA/SVD or
cross-modal subspace overlap require new collection. Testing a new pruning or
cache policy's correctness also requires actual model execution. Existing final
answers can be rescored without generation; intermediate policy reasoning and
per-policy commitment trajectories are not archived. The full baseline commit
history is archived and, because this decoder does not remask committed tokens,
its token states can be reconstructed using the final baseline token IDs.

Raw audio is referenced by path/source/hash, not copied into the run directory.
Preserve the manifest and audio files as well as the whole run and console log
if a self-contained input archive is needed. Dataset/model provenance helps
re-download inputs but is not a substitute for the original bytes.

`DIAGNOSTIC_QUERIES=16` is an optional broader answer-query sample before a new
run; it increases diagnostic work and is still not exhaustive. Query selection
and layer coverage must be chosen before collection. Thresholds, horizons,
histories, progress bins, layer groupings and final-answer parsing can be changed
afterward. The added high-to-low report uses the same schema-2 recordings and
requires no additional GPU forwards.
