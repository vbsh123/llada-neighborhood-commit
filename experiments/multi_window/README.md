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

## Exact first-version choices

1. The first forward is global: initialize every layer's cache and choose the
   ordinary global top1 commitment (or threshold batch). No initial prefix window
   is imposed. Then open a window centered on every commitment.
2. Default radius is **4 response token positions on each side**, an inclusive
   width of at most 9, clipped at sequence ends. This is positional distance,
   **not a count of MASKs**, and is unrelated to the old region-analysis gap rule.
3. On local forwards, every row inside the union of windows is recomputed,
   including filled tokens. Only masked rows there receive vocabulary predictions.
   Normal top1/threshold selection runs on those fresh predictions. No neighbor
   forcing, confidence boost, entropy gate, or stale-logit commitment.
4. Each commitment adds `[position-radius, position+radius]` to the union. A window
   therefore expands when its commitments approach an edge. Overlapping or
   directly touching intervals merge; even one uncovered position keeps intervals
   separate. There is no maximum active-row/window budget in this first version.
5. A full refresh occurs at steps **0, 8, 16, ...** by default. All response rows
   recompute and all remaining MASKs compete. Fresh distant commitments can open
   new windows. A global refresh itself does not open windows around unselected
   predictions. Threshold batches may seed several windows at once.
6. Retire a window before a forward if it contains no remaining MASKs. If no
   window can supply a candidate, do an extra global refresh. Those extra
   refreshes do not reset the periodic cadence. There is no inactivity timeout.
7. Response positions outside windows really sleep, **including filled positions**:
   no fresh QKV, query attention, FFN, or vocabulary projection for them. Awake
   queries can attend to cached K/V at **all original sequence positions**.
   Prompt positions stay fresh. A newly committed response row always wakes on
   the next forward, even if its window retired, so its cache incorporates the
   actual revealed token rather than indefinitely retaining MASK K/V.
8. This differs from upstream Window-Diffusion's external-context restriction:
   we preserve global cached attention context. It also differs from the previous
   entropy experiment, where all filled response rows always recomputed. These
   choices should be treated as experimental, not proven safe.
9. Default decoder is one-token top1, temperature zero, full 256 response slots.
   Threshold policy is optional: every fresh eligible candidate >= threshold,
   falling back to its highest-confidence fresh candidate. A local top1 is not
   claimed to be the global top1 between refreshes.
10. Prompt is the same question-only chat template and pinned model as before;
    no extra instruction, hidden block constraint, EOS suppression, or early
    stopping change. Fill the entire response canvas, then truncate the saved
    answer at its first final stop token. Post-stop work contributes to runtime.

Radius and refresh interval are initial engineering choices, not fitted optima.
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

ALLOW_SCALED_FP32_DRIFT=1 RUN_NAME=multi_window_smoke_v1 \
bash scripts/vast_multi_window.sh smoke

ALLOW_SCALED_FP32_DRIFT=1 SKIP_INSTALL=1 RUN_NAME=multi_window_pilot_v1 \
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
RUN_NAME=multi_window_matched_v1 bash scripts/vast_multi_window.sh pilot
```

For ordinary vanilla too, add `INCLUDE_ORIGINAL=1` (separate opt-in). No run here is
presented as a single-window Window-Diffusion baseline; that comparison remains
future work. Earlier timings from another machine are reference values only.

Example settings / same saved GSM8K questions:

```bash
RADIUS=4 REFRESH_EVERY=4 POLICY=threshold COMMIT_THRESHOLD=0.9 \
SAMPLES_FROM=runs/multi_window_pilot_v1/samples.jsonl \
ALLOW_SCALED_FP32_DRIFT=1 SKIP_INSTALL=1 RUN_NAME=multi_window_threshold_v1 \
bash scripts/vast_multi_window.sh pilot
```

`REFRESH_EVERY=1` is an all-response-awake scheduling control. Larger windows or
frequent refreshes may remove most savings. Small windows / infrequent refreshes
may miss distant ready tokens and freeze useful context. There is no speed or
accuracy claim until GPU measurements.

## Results and analysis

Download `runs/RUN_NAME/` or the printed `exports/*.tar.gz` and checksum. It includes:

- `comparison.json`: accuracy, forwards, total decode/model seconds, row counts,
  sleep fraction, summed refresh/open/retire/merge counts, maximum window count,
  and full-response-slots/sec. Speedup appears only when a baseline was requested.
- `multi_window/samples/ID/result.json`: output, scoring, timing and window stats.
- `multi_window/samples/ID/trace.jsonl.gz`: per-forward windows before/after,
  refresh reason, sleeping rows (may include filled slots), commitment positions,
  IDs/confidences, and best fresh masked confidence in each pre-batch window.
  `best_outside_window_confidence` is only observable on full refreshes; null on
  local forwards. No complete per-position probability vectors are stored.
- Manifest: pinned checkpoint, source fingerprint, dataset revision / sample hash,
  seed, settings and GPU. Resume skips completed samples and rejects mismatches.
- Validation and optional layer diagnostics, packages and nvidia-smi capture.

`windows_opened` counts resulting interval components with no old-window parent;
`new_region_seeds` counts commitments outside pre-batch intervals, which may merge
immediately. `windows_merged` counts old components absorbed together (k parents
means k-1 mergers); retirement is separate. Window-count sum is over pre-forward
windows, max includes post-batch windows. Global refreshes are ordinary measured
generation forwards, not extra unreported diagnostic passes.

Timing excludes checkpoint loading, warmup, numerical validation and serialization;
includes scheduling, scoring, global refreshes and cache updates. Vocabulary
entropy is computed by the shared scorer but does not affect window scheduling.

Established extraction on saved answers, without generation:

```bash
source .venv/bin/activate
python -m experiments.gsm8k_eval --run runs/multi_window_pilot_v1
```

This reuses harness extraction, not the harness's five-shot prompting protocol.
