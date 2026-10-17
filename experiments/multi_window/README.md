# Multiple active computation windows

This is a separate experiment, inspired by Window-Diffusion's active computation
and cached context. It uses our existing pinned LLaDA-Instruct selective backend,
not the upstream generator. It does not force neighboring tokens to commit.

## Read the code

- `schedule.py`: `prepare` decides which response rows sleep **before** a forward;
  `finish` adds neighborhoods around this forward's commitments and unions the
  intervals. `merge_windows` implements interval union.
- `run.py`: connects that scheduler to the shared runner. Default: `multi_window`
  only. `--compare-active` and `--include-original` explicitly add baselines.
- `../token_sleep/selective.py`: actual gather/scatter QKV, original-position RoPE,
  full cached attention context, and active-row FFN computation.
- `../token_sleep/benchmark.py`: shared generation, validation, timers, scoring,
  manifests and resume checks. Optional factory injection preserves the existing
  entropy-sleep runner's defaults.
- `../../tests/test_multi_window.py`: CPU scheduling and toy-model integration.
- `../../scripts/vast_multi_window.sh`: Vast setup, tests, generation, export.

## Corrected region definition (v2)

Every filled response token belongs to a persistent region. Two groups are
separate iff **at least REGION_GAP consecutive MASKs** lie between them (default
4). A region includes its complete span: old commitments, new commitments, and
any intervening gaps shorter than REGION_GAP.

Example with gap=4: filled positions 10 and 14 have three MASKs between them, so
form region [10,14]. Filled positions 10 and 15 have four MASKs between them,
so form two regions until a bridging commitment reduces the gap.

The computation window for EACH region [left,right] is
`[left-RADIUS, right+RADIUS]`, clipped to the response boundaries. Default RADIUS=4.
Thus a region [10,40] has window [6,44], not merely a small window around its most
recent token. All regions remain awake regardless of how long ago they received
commitments. **There is no retirement or inactivity timeout.**

These are two independent parameters:

- REGION_GAP: consecutive MASKs defining logical region separation.
- RADIUS: extra positional padding outside each region, default four on either side.

Padded windows may overlap even while logical regions are separate. We union
these windows for efficient execution, but only the MASK-gap rule merges logical
regions. Traces and counters record logical regions and physical window unions
separately. Padding overlap is not reported as a logical region merger.

1. First forward is global to initialize all per-layer caches. Ordinary global
   top1 (or a threshold batch) seeds the first region(s), with no prefix constraint.
2. Before every subsequent forward, rebuild regions from ALL currently filled
   response positions, then pad every region and union computation intervals.
3. Recompute every row in that union, including every filled response position
   and all interior MASKs. Only awake MASK rows receive vocabulary predictions.
   No forced neighbor commitments, confidence boosts, or entropy scheduling.
4. Refresh all response positions at steps 0,4,8,12,... by default. Select among
   all remaining MASKs on those forwards; distant commitments can create regions.
   Otherwise selection sees only freshly evaluated MASKs inside computation windows.
5. If the window union supplies no masked candidate, do an additional global
   refresh without retiring any region or resetting the periodic cadence.
6. Only MASK positions outside the windows sleep. Their cached K/V stays available
   globally to awake queries. Filled response rows and prompt rows always stay
   fresh. This retains the filled-token freshness of the original sleep experiment.
7. There is no maximum window-count/active-row budget. As regions grow and merge,
   active computation can approach the full response. This is intentional under
   the requested persistent-region rule, not an established optimum.
8. This uses the pinned LLaDA-Instruct checkpoint and preserves global cached
   attention context; it is inspired by, not a reproduction of, Window-Diffusion.
9. Decoder defaults to one-token top1, temperature zero. Optional threshold policy
   commits every fresh candidate >= threshold, with a fresh-top1 fallback. Local
   top1 is not claimed to be global top1 between refreshes.
10. Same question-only chat template and 256-slot full-response generation as the
    sleep runs: no extra instruction, hidden block constraint, EOS suppression,
    or early stopping change. Final text is truncated at the first final stop;
    post-stop computation remains included in runtime and region construction.

Version v1 used commitment-centered interval growth and retired exhausted windows.
It did NOT implement the clarified persistent MASK-gap region definition. Use new
v2 output folders: source/config fingerprints prevent mixing these experiments.

Padding and refresh interval are initial engineering choices, not fitted optima.
The compatibility fields `high_entropy`, `high_sleep`, `sleep_confidence` in the
shared manifest are unused by this scheduler; its CLI does not expose them.

## Upstream inspection and attribution

Inspected the official repository, cloned read-only to a temporary directory:
https://github.com/vhicrgit/Window-Diffusion

Pinned inspected commit: `d8bb349e81b74741900049f307184e848006b330`.
Relevant source: `llada/model/modeling_llada.py`, especially `build_window_plan`,
`update_window_plan`, `block_forward_prune` and `generate`. These separate active
rows from a larger context window and alternate full and pruned execution.

No upstream source is vendored or copied into this experiment; the implementation
uses our already tested selective backend and new interval scheduling. This is
not a reproduction of upstream Window-Diffusion or a novelty claim.

## Full commands on a new Vast machine

Use a BF16-capable CUDA GPU and Python 3.10–3.12. No separate HF cache exports are
required. Download interruptions may require rerunning the same command.

```bash
cd /workspace
git clone --branch sleep-simulation https://github.com/vbsh123/llada-neighborhood-commit.git
cd llada-neighborhood-commit

ALLOW_SCALED_FP32_DRIFT=1 RUN_NAME=multi_window_smoke_v2 \
bash scripts/vast_multi_window.sh smoke

ALLOW_SCALED_FP32_DRIFT=1 SKIP_INSTALL=1 RUN_NAME=multi_window_pilot_v2 \
bash scripts/vast_multi_window.sh pilot
```

On an existing checkout, `git pull --ff-only` first. Smoke = **2 actual GPU
questions**, pilot = **10**, study = **100**. No vanilla or active baseline is
run unless explicitly requested. Use a new RUN_NAME when changing source/settings.

The opt-in `ALLOW_SCALED_FP32_DRIFT=1` uses the same existing numerical validation
route as the earlier sleep experiments. Actual generation remains BF16. It does
not convert generation to FP32 or waive every failure: dense/sparse checks,
selection checks and local FP32 math checks still gate execution. Strict BF16
parity failure remains recorded. Default without that environment setting is
strict validation; BF16 changes in operation shape can cause it to stop.

For a matched comparison **only when wanted**, explicitly add the dense selective
baseline, which uses the same backend and fresh-mask-only vocabulary head:

```bash
COMPARE_ACTIVE=1 ALLOW_SCALED_FP32_DRIFT=1 SKIP_INSTALL=1 \
RUN_NAME=multi_window_matched_v2 bash scripts/vast_multi_window.sh pilot
```

For ordinary vanilla too, add `INCLUDE_ORIGINAL=1` (separate opt-in). No run here is
presented as a single-window Window-Diffusion baseline; that comparison remains
future work. Earlier timings from another machine are reference values only.

Example settings / same saved GSM8K questions:

```bash
REGION_GAP=4 RADIUS=4 REFRESH_EVERY=4 POLICY=threshold COMMIT_THRESHOLD=0.9 \
SAMPLES_FROM=runs/multi_window_pilot_v2/samples.jsonl \
ALLOW_SCALED_FP32_DRIFT=1 SKIP_INSTALL=1 RUN_NAME=multi_window_threshold_v2 \
bash scripts/vast_multi_window.sh pilot
```

`REFRESH_EVERY=1` is an all-response-awake scheduling control. Larger windows or
frequent refreshes may remove most savings. Small windows / infrequent refreshes
may miss distant ready tokens and freeze useful context. There is no speed or
accuracy claim until GPU measurements.

## Results and analysis

Download `runs/RUN_NAME/` or the printed `exports/*.tar.gz` and checksum. It includes:

- `comparison.json`: accuracy, forwards, total decode/model seconds, row counts,
  sleep fraction, summed refresh/open/merge counts, maximum region/window counts,
  and full-response-slots/sec. Speedup appears only when a baseline was requested.
- `multi_window/samples/ID/result.json`: output, scoring, timing and window stats.
- `multi_window/samples/ID/trace.jsonl.gz`: per-forward regions and padded windows before/after,
  refresh reason, sleeping rows (MASKs only), commitment positions,
  IDs/confidences, and best fresh masked confidence in each pre-batch window.
  `best_outside_window_confidence` is only observable on full refreshes; null on
  local forwards. No complete per-position probability vectors are stored.
- Manifest: pinned checkpoint, source fingerprint, dataset revision / sample hash,
  seed, settings and GPU. Resume skips completed samples and rejects mismatches.
- Validation and optional layer diagnostics, packages and nvidia-smi capture.

`regions_opened` counts post-batch logical components with no pre-batch parent;
`regions_merged` counts old logical components absorbed together (k parents means
k-1 mergers). `new_region_seeds` counts commitments in parentless new regions.
`windows_opened` and `windows_merged` apply the same bookkeeping to the padded
interval union; these are NOT logical-region counts. Count sums use pre-forward
geometry, maxima also include post-batch geometry. `windows_retired` and
`sleeping_filled_position_forwards` remain zero by construction.
Global refreshes are ordinary measured generation forwards, not extra unreported
diagnostic passes.

Timing excludes checkpoint loading, warmup, numerical validation and serialization;
includes scheduling, scoring, global refreshes and cache updates. Vocabulary
entropy is computed by the shared scorer but does not affect window scheduling.

Established extraction on saved answers, without generation:

```bash
source .venv/bin/activate
python -m experiments.gsm8k_eval --run runs/multi_window_pilot_v2
```

This reuses harness extraction, not the harness's five-shot prompting protocol.
