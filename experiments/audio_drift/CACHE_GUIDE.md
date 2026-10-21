# Reading the DIFFA cache experiment

Start with `decode.py`, then `compact.py`, then `protocol.py` and `benchmark.py`.
The implementation changes inference only; it does not modify the pinned upstream
checkout, retrain weights, prune audio positions or skip answer-token updates.

## 1. Ordinary generation: decode.py

`Prepared` identifies acoustic, semantic and fixed prompt positions in the full
sequence. `forward_logits` embeds the current text token IDs and inserts the
already-computed audio embeddings. `decode` repeatedly calls the transformer;
`commit` fills the scheduled number of high-confidence answer positions in the
current block. All transformer answer rows remain active, including positions
whose discrete token ID has already been committed.

The audio encoder and connectors run once in `model.py:prepare`. Their output
embeddings are fixed. The deeper transformer representations can still change
because bidirectional attention sees a changing answer.

## 2. Actual compute skipping: compact.py

`PrefixCache` temporarily replaces each transformer block's forward method.
It restores the original methods on context exit, including exception paths.
Each layer has its own full-position K, V and block-output hidden-state banks.
Keys are cached after positional rotation. This is more than a KV-only cache:
the saved block outputs let inactive prefix positions skip their attention
queries and MLP work too.

The first forward computes every row and initializes every bank. On later
forwards, a layer independently determines which prefix groups are due for
refresh. Active rows include all answer positions and any refreshing prefix
group. The wrapper then:

1. Gathers those rows from the incoming hidden matrix.
2. Computes their normalization, Q/K/V projections and absolute-position RoPE.
3. Replaces only their slots in the full K/V banks.
4. Computes attention from active queries to **all** cached/fresh keys and values.
5. Runs the attention output projection and MLP only for active rows.
6. Writes their block outputs into the full hidden bank. Other rows retain their
   cached output from that layer.

Audio K/V positions remain visible. The attention score matrix is active rows
by all key positions, rather than all rows by all positions. Embedding, final
normalization and vocabulary projection still process the full sequence equally
for baseline and cached runs.

A deep layer can refresh audio while a shallower layer reuses its audio output.
The deep refresh then uses that cached shallow output. It does not use an oracle
future baseline representation. The dense-reference test explicitly checks this
case by freezing both the inactive K/V slots and inactive block outputs.

## 3. Refresh clocks: protocol.py

`conditions` constructs flat schedules. `with_depth_variants` pairs a flat policy
with a variant using the same multipliers as every other policy, including the
shared-prefix control. `layer_intervals` expands equal-sized layer groups into
an interval per transformer block.

The overnight multipliers are `[8,4,2,1]` over layers `[0-7,8-15,16-23,24-31]`.
For the base A16/P8/S4 policy:

| Layers | Acoustic | Prompt | Semantic |
|---|---:|---:|---:|
| 0-7 | 128 | 64 | 32 |
| 8-15 | 64 | 32 | 16 |
| 16-23 | 32 | 16 | 8 |
| 24-31 | 16 | 8 | 4 |

A positive interval refreshes on forward 0 and multiples of that interval.
Zero initializes on forward 0 and never refreshes afterward. Nested intervals
ensure a shallow refresh for a group also refreshes that group at deeper layers.
The deepest layer group's interval matches the flat policy. The resulting
schedule is an experimental choice informed by drift, not an optimal interval
calculated from cosine distance.

`PrefixCache.accounting` sums actual row updates across all layers. A skipped-row
fraction is not a speedup or a complete FLOP count; measured timings are separate.

## 4. Timings and safeguards: benchmark.py and replay.py

`benchmark` runs independent, autonomous generations for quality and latency.
It uses one warmup and two timed repeats per policy, with fresh caches for each
pass and synchronized GPU wall timing. A clean baseline brackets the policy
measurements. Identity controls must reproduce baseline visible token IDs;
timed repeats must be reproducible. Output lengths and forward counts are saved.
The overnight policy order is shuffled deterministically per sample, keeping the
identity first. This reduces fixed-order bias; recorded policy orders permit audit.

`benchmark_replay` supplies the same saved baseline token states to every policy
for the same number of initial pre-EOS forwards. It does not commit model
predictions or grade replay outputs. It isolates forward-path savings from
changes in generated answer length. It is not end-to-end generation latency.

`hooks.py` contains the earlier dense KV substitution diagnostic: it computes
fresh rows and then substitutes cached K/V. Read **compact.py** for the mechanism
that actually saves computation. Dense continuation branches do not support the
new depth policies; the overnight preset uses compact generation and replay.

## 5. Verification and remaining limits

The CPU suite uses the real pinned upstream transformer at small dimensions.
It checks flat-cache equivalence, depth-cache equivalence against an independent
full-row oracle, absolute positions, per-layer accounting, answer rows staying
active, synchronization and restoration of original model methods. Integration
tests cover generation, replay, cohort reporting and the hook-free runner.
The combined-manifest test excludes development source IDs from fresh selection
and pins fresh examples to the same dataset revision.

```bash
DIFFA_ROOT="$PWD/external/DIFFA" .venv-audio/bin/python \
  -m unittest discover -s tests -p test_audio_drift.py -q
```

CPU correctness tests are not a substitute for full-model BF16 GPU validation.
The Vast run checks official baseline decoding on its first two examples and
identity-cache outputs on every example. Different matrix shapes can cause
numerical differences for approximate policies; their own generated answers,
not dense-diagnostic answers, are scored. Raw outputs, strict scores, normalized
scores, repeated timings and memory/accounting remain available for inspection.


Delayed KV-column pruning and the 32/50 refresh experiment are described in
[PRUNING_GUIDE.md](PRUNING_GUIDE.md). Existing cache-only behavior is unchanged
when no pruning specification is supplied.
