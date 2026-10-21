# Delayed audio/text pruning and longer cache intervals

`gsm-prune` reuses the current spoken-GSM manifest. This is an exploratory
quality/latency experiment, not held-out validation. It collects final answers,
strict/normalized numerical scores, clean autonomous timings and matched-input
replay. It does not repeat the large baseline drift/attention archive. The fixed
quota decoder, 512 answer positions, 256 maximum forwards and safe EOS stopping
remain unchanged. All examples ending before a pruning anchor stay in primary
results; the activation report exposes them separately.

## Six cache-only conditions

- Refresh every forward (identity).
- Shared prefix interval 8.
- Shared prefix interval 32 = A32/P32/S32.
- Shared prefix interval 50 = A50/P50/S50.
- A32/P8/S32.
- A50/P8/S50.

Intervals count consecutive global model forwards, including across decoding
blocks, with initialization at index zero. Interval 50 deliberately is not
aligned to the 16-forward decoding blocks. A 50 interval has no later refresh
on an example that ends within 50 forwards; actual execution counts are saved.

## Twelve pruning conditions by default

Every strategy is paired across three eligible pools: audio only, fixed prompt
text only, and the full conditioning prefix (audio plus text). Answer positions
are never candidates. Each corresponding comparison has the same absolute
removal budget, anchor, layer range and base caching schedule.

- Shared-8 + lowest-mean-attention pruning after 16 forwards: three pools.
- Shared-8 + lowest-mean-attention pruning after 32 forwards: three pools.
- Shared-8 + random pruning after 32 forwards: three pools.
- Refresh-every-forward + lowest-mean-attention pruning after 32: three pools.
  This last trio isolates pruning without prefix cache reuse.

Removal count is floor(0.25 * number_of_audio_positions), capped to leave at
least one position in each of the audio and prompt groups. This is a
COUNT-matched experiment, not a fraction-matched experiment: the much shorter
prompt can lose a large fraction of its positions. Actual count/fraction and
removed acoustic/semantic/prompt counts are logged. `PRUNE_FRACTIONS` changes
the requested audio-based quota. Structural prompt positions are not exempted;
interpret fixed-template text pruning accordingly.

Selection uses the state at the anchor in layer 24, averaged across all heads
and 16 evenly spread answer-CANVAS query positions, including masks and future
blocks. It does not use the final answer length, a baseline trajectory, future
attention, correctness labels, or retrospective revealed fractions. Attention
is reconstructed in FP32 for selection; the ordinary model attention backend
is used for generation. Random variants receive reproducible sample-specific
seeds and record the same attention-mass diagnostic for overhead comparability.
This first intervention ranks instantaneous mean attention. It does not claim
a persistent-low or peak/drift-guarded safety rule.

The same selected positions are omitted from attention KV columns immediately
at layer 24 and in all deeper layers. They stay omitted on later forwards,
including refresh forwards. Their QKV/query/MLP rows are skipped from the next
forward in those layers; selection happens after anchor QKV computation.
Earlier layers still process these positions normally under the base cache.
Original RoPE coordinates remain unchanged. No removed lower-layer position is
revived in a deeper layer using a stale hidden state.

Full-size backing KV/hidden banks and full embedding/final norm/vocabulary
allocations are retained; retained KV columns are gathered before attention.
Thus this measures real attention-column and transformer-row work reduction,
including selection/gather overhead, not GPU memory reclamation or guaranteed
speedup. Both peak allocated memory and actual row/KV/query-key-pair counts are
saved. Cached information can still reach answer positions through earlier
layers; this is deliberate late-layer removal.

## Run

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_gsm_prune20_v14 SAMPLES=20 SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh gsm-prune \
  > audio_gsm_prune20_v14.log 2>&1 &
tail -f audio_gsm_prune20_v14.log
```

Default total: uncached baseline plus 18 benchmark policies on each example,
one policy warmup plus two timed repeats, and two matched-input replay repeats
through up to 256 baseline pre-EOS forwards. All selection and gathering costs
are included in both autonomous timing and replay. Replay outputs are not
scored; replay and autonomous runs may select different positions because they
follow different token trajectories. There is no repeat of full attention
histories/operator profiling in this mode. `SAMPLES=1` with a new run name is a
GPU smoke option before committing to the full experiment.

`PRUNE_STARTS="16 32"`, `PRUNE_LAYER_START=24`, `PRUNE_QUERIES=16`,
`PRUNE_FRACTIONS="0.25"`, `BENCHMARK_REPEATS`, `REPLAY_REPEATS`,
`REPLAY_SAMPLES` and `REPLAY_STEPS` are adjustable. Random/no-cache controls are
added at the largest requested anchor; attention/shared-8 pairs cover every
anchor. Multiple requested fractions expand every corresponding scope pair.
Pruning is tested on shared-8 and without cache reuse, not crossed with every
cache interval. The longer intervals form a separate cache-only comparison.

## Read the results

- `cache_benchmark.json`: policy specs, final texts/token IDs, quality, repeats,
  peak memory and exact execution accounting. Pruning accounting records anchor,
  absolute positions, pool composition, removed mass and downstream layers.
- `fixed_input_replay.json`: same baseline inputs and forward count for each
  policy, separate initialization/selection in each fresh run.
- `analysis/reasoning_report.md`: primary paired quality and speed results.
- `analysis/pruning_report.md` and `pruning_pairs.csv`: activation counts,
  actual budgets/fractions, group composition and per-example compute/timing.
  Activation is diagnostic; do not drop unactivated/failed baseline examples
  from primary scores to improve the headline result.

Manual review plus pinned tiny-upstream FP32/BF16 tests check agreement with an
independent reference that computes every row and explicitly removes the same
KV columns, delayed onset, retained RoPE positions, downstream mask persistence,
identity controls, scope pairing, measured row/KV counts, repeat reproducibility
and report integration. Full-size CUDA speed and quality require the GPU run.

For a fresh machine reproducing v13, set `WHISPER_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1`.
Recreate the development manifest with seed 1729 at GSM8K-Speech revision
`e94caa2154433a54cd430702666c454977df3083`, then recreate the 30-row combined
manifest using seed 20261020 and that base manifest. The first 20 rows reproduce
the preceding diagnostic cohort; these remain exploratory rather than held out.
