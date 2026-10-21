# Separate cache and delayed-pruning experiments

Test each idea independently before combining them. Both modes reuse the first
30 rows of the existing spoken-GSM manifest, retain baseline failures, save final
answers/scores/forward counts, and measure autonomous decoding plus identical-input
replay. They omit repeated large drift/attention collection and operator traces.
These are exploratory examples, not held-out validation.

## Cache only: gsm-cache

Exactly four conditions, acoustic / fixed prompt text / semantic:

- A32/P8/S32: prior promising reference, remeasured on the new GPU.
- A32/P32/S32: shared-32 sanity comparison.
- A50/P8/S50: longer audio caching while keeping prompt at 8.
- A50/P50/S50: shared-50 comparison.

No identity policy, shared-8 policy, or pruning. A fresh uncached baseline is
still timed for quality/latency comparison on the current machine. Intervals
count global consecutive forwards across decoding blocks, initializing at zero.
Interval 50 is intentionally not aligned with the 16-forward decoding blocks;
examples ending within 50 forwards only initialize that cache.

## Pruning only: gsm-prune

Twenty policies; no cache reuse for surviving positions:

| Selection | After how many forwards? | Eligible pools |
|---|---:|---|
| Lowest mean attention | 1 (early control) | Acoustic; semantic; combined audio; text; full prefix |
| Lowest mean attention | 16 | Same five pools |
| Lowest mean attention | 32 | Same five pools |
| Random, matched budget | 32 | Same five pools |

Across the five pools, timing/layers and absolute removal counts match. Count
is floor(0.25 * audio positions), capped to leave at least one position in each
of the acoustic, semantic and text groups. With 64 acoustic positions the common
budget can therefore be at most 63. This matches counts, not fractions: the short text
prompt may lose a large fraction of its positions, including structural tokens.
Actual counts, fractions and acoustic/semantic/prompt composition are recorded.

Anchors use only observed forward number. With 512 answer slots and two commits
per forward, the defaults correspond to 2/32/64 revealed slots
(0.39%/6.25%/12.5% of the full canvas). These are NOT fractions of the unknown final answer length.
Autonomous selection records the actual remaining masks, revealed fraction,
block and block progress. Forward index and global canvas unmasking are coupled
under the fixed-quota decoder; they are not independent triggers here.

Selection happens once at layer 24, using all heads and 16 fixed answer-CANVAS
query positions, including masks and future decoding blocks. No baseline future
trajectory, final answer span, future attention, or target answer is consulted.
The FP32 selection softmax ranks current mean attention; this is a quota test,
not a calibrated persistent-low/peak/drift-guarded safety rule. Random controls
use deterministic sample-specific seeds and record removed attention mass too.

Selected KV columns disappear immediately in layers 24–31 and remain omitted
on later forwards. Their QKV/query/MLP rows are skipped from the next forward in
those layers. The SAME positions stay removed throughout the deeper stack;
a removed lower-layer position is never revived with stale representations.
Earlier layers and all surviving rows recompute every forward, with unchanged
absolute RoPE coordinates. All answer rows remain active. Full-size backing
banks, embeddings, final norm and vocabulary projection allocations remain;
retained KV columns are gathered before attention. This is real column/row
work removal with selection/gather overhead included, not memory reclamation
or a guaranteed speedup. The adapter's internal interval 1 means no reuse of
surviving prefix representations. Deep-layer-only pruning may give modest gains.

## Run sequentially on the same GPU

Use a new run name for each stage. Install and prepare the pinned manifests
first (the setup steps from the previous instructions remain valid).

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env SAMPLES=30 SKIP_INSTALL=1 REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  WHISPER_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1 \
  bash -c '
    RUN_NAME=audio_gsm_cache30_v16 bash scripts/vast_audio_drift.sh gsm-cache > audio_gsm_cache30_v16.log 2>&1 &&
    RUN_NAME=audio_gsm_prune30_v16 bash scripts/vast_audio_drift.sh gsm-prune > audio_gsm_prune30_v16.log 2>&1
  ' > audio_followup_v16.log 2>&1 &
```

The pruning stage starts only if the cache stage succeeds. Download BOTH run
folders and both stage logs. Full-size checkpoints download on first use.
Both stages use 512 answer slots, 256 maximum forwards, 32-position blocks and
safe EOS stopping. Each policy has a warmup and two timed repeats, plus two
identical-input replay repeats up to the complete baseline pre-EOS trajectory.
Replay outputs are not graded, and its token selections can differ from those
in autonomous generation. Full-size CUDA execution remains to be validated;
CPU tests cover the pinned tiny upstream model in FP32/BF16 and explicit KV
removal, scope pairing, onset/propagation, accounting, repeat behavior and reports.

## Retained data and controls

`cache_benchmark.json` saves final texts/IDs/scores, schedules, time repeats,
peak allocated memory, actual row/KV/query-key-pair accounting, and selected
positions/group composition/removed mass. Pruning anchors also record actual
autonomous unmasking progress. `fixed_input_replay.json` measures matched-state
forward timing independently of generated length; all selection/gather costs
are included. `analysis/reasoning_report.md` reports paired quality/latency.
The pruning stage additionally writes `pruning_report.md` and `pruning_pairs.csv`,
including activation and actual budgets. Examples ending before their pruning
anchor remain in primary scores; inspect activation separately.

`PRUNE_STARTS="1 16 32"`, `PRUNE_FRACTIONS="0.25"`, `PRUNE_LAYER_START=24`,
`PRUNE_QUERIES=16`, `BENCHMARK_REPEATS`, `REPLAY_REPEATS`, `REPLAY_SAMPLES`,
`REPLAY_STEPS` and `SAMPLES` are configurable. Random controls are added at the
largest requested anchor; every strategy is paired across all five pools.

For reproducibility, the previous dataset revision is
`e94caa2154433a54cd430702666c454977df3083`: prepare 10 development examples
with seed 1729, then the 30-row combined manifest with that base and seed
20261020; run all 30 rows. The LLaDA and DIFFA adapter revisions are
already pinned; the command above pins the previous Whisper revision too.

The cache and pruning runs now default to 30 examples. The early pruning
control, atomic-group removal and matched budgets are intentional additions;
use a new run name when changing these settings.
