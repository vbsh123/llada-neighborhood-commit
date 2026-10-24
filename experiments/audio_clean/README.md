# Clean audio experiments (v27)

This directory is the new entry point. The old experiments remain archived in
`audio_drift`; this runner has no drift collection, boosting, replay, parameter
sweeps or repeated per-example timings. It imports the pinned model/input loader,
unchanged decoder/scorers, tested prefix cache, and physical compaction engine.
It does not import the old `audio_drift.run` orchestration.

All three suites use the **exact earlier 50 questions**, validated by the frozen
input fingerprint `8ed78386500a79daecdf9b9be573efe2c2a4001a5c667b23c58555a6e03205f2`.
These are development data. `SAMPLES=2` selects the first two *after* verifying the
complete 50-question fingerprint. No silent new dataset or decoder settings.

## Three separate suites

| Suite | Baseline plus policies |
|---|---|
| `cache` | 8/8/8, 16/8/16, **16/16/16** |
| `prune` | Global-importance audio pruning and matched-count/time generic prefix pruning |
| `sparse` | Sparse audio alone, ordinary cache8/8/8, sparse audio + cache8/8/8 |

All use 512 answer slots, block32, maximum256 forwards and safe EOS stopping.
Commitment is the unchanged fixed-quota, top-confidence decoder (2 positions per
forward), not a confidence-threshold early-commit variant. Every answer position
is recomputed on each forward. Caching affects fixed-prefix rows only.

### Global pruning

Default `QUERY_SCOPE=all` measures importance over **every surviving query row**:
audio, fixed prompt, committed answer, current and future MASK answer positions.
For each layer/head/query, compute softmax over all retained keys; average heads
and queries, normalize per key against that layer's uniform `1/key_count`, then
average all32 layers. Actual self-attention is included. This is global incoming
attention importance, not just total attention mass assigned to an audio group.

`QUERY_SCOPE=all_masked` instead measures all remaining answer MASK queries across
all answer blocks. `QUERY_SCOPE=current_block` reproduces the old query scope.
`COMPARE_CURRENT_BLOCK=1` adds an old-scope audio/generic pair in the same run.
This isolates the query choice while retaining the old decoder and pruning rule.

Keep the previous gentle development setting: normalized score <0.5 for two
observations; observe every8 forwards; allow acoustic from zero-based16, semantic
from44; remove the bottom12.5% of qualified surviving positions per group/check,
rounded up. No fixed token cap. Changes apply next forward. All-layer removal by
default; `PRUNE_LAYER_START=25` limits physical removal to25–31 while still scoring
all32 layers. This is not a newly tuned threshold.

The generic control follows the audio policy's event times and removal counts,
but ranks its own surviving *entire* fixed prefix. It is reference-guided,
count-matched, and is not independently optimized. It does not have to satisfy
the audio threshold. Never interpret it as an independently deployable detector.

Exact global scores require extra QK work. Queries are chunked32 at a time to
bound memory, with no query subsampling. Observation/selection cost is included
in decode timing: a more informative detector may cost more than it saves.

### Audio sparse attention

Paper-based adaptation of [MARS-Cache §4.2](https://arxiv.org/html/2601.17868v2#S4.SS2).
The public VidLaDA checkout inspected for this work does not contain its MARS
attention kernel. This is **not an exact reproduction**, nor an established
audio-locality finding.

* First forward: ordinary full attention. Select semantic anchors once at the
  first layer of each sparse layer group, using128 evenly spaced queries over
  the whole sequence, mean/summed over heads, semantic-key softmax with diagonal
  self entries masked. Cache indices for the remaining decode.
* Semantic audio: chronological chunks of32 projected semantic tokens, with own
  and adjacent chunks available to non-anchor semantic queries.
* Layers0–7 remain dense. Groups8–15,16–23,24–31 keep respectively50%,25%,12.5%
  semantic anchors **per chunk**, rounded up, minimumone. These are conservative
  engineering starting settings, **not parameters validated by our drift data**.
* Acoustic summary queries and semantic-anchor queries retain full attention.
  Prompt and every answer query see all non-semantic keys plus semantic anchors.
  Other semantic queries also see their local semantic neighborhood. Thus text
  can access distant non-anchor semantic information indirectly through anchors.
* Acoustic tokens remain globally accessible. They are learned summary slots;
  we do not assume their index is chronological. No tokens are physically deleted.
* Separate gathered Q/K/V subsets feed SDPA. We avoid computing forbidden pairs
  rather than applying a full dense NxN mask. Original RoPE coordinates remain.

Reported pair counts exclude anchor-search work; decode timing includes it and
every gather/scatter/kernel launch. Many short SDPA calls may be slower on an A40.
This requires a GPU smoke check before committing to50 examples. Sparse-only
isolates the new attention pattern; sparse+cache versus ordinary cache isolates
the additional effect. There is no simultaneous token pruning in this suite.

## Launch on a new Vast machine

```bash
cd /workspace
git clone --branch audio-drift-study https://github.com/vbsh123/llada-neighborhood-commit.git llada-audio-clean
cd llada-audio-clean
# Installs dependencies and prepares the exact earlier question pool.
nohup env SKIP_INSTALL=0 bash scripts/vast_audio_clean.sh cache > audio_clean_cache50_v27.log 2>&1 &
tail -f audio_clean_cache50_v27.log
```

After setup, independently select pruning or a two-question sparse pilot:

```bash
nohup env SKIP_INSTALL=1 COMPARE_CURRENT_BLOCK=1 bash scripts/vast_audio_clean.sh prune > audio_clean_prune50_v27.log 2>&1 &
nohup env SKIP_INSTALL=1 SAMPLES=2 bash scripts/vast_audio_clean.sh sparse > audio_clean_sparse2_v27.log 2>&1 &
```

Run only **one job per GPU at a time**. These commands are alternatives, not a
request to start all three together. For the smallest pruning pilot also set
`SAMPLES=2`. To run sparse50 after inspecting its smoke output, remove `SAMPLES=2`.
Never pull into a checkout while its job is active.

## Outputs and review

`runs/audio_clean_<suite><N>_v27/analysis/report.md` reports accuracy, accuracy
change in percentage points, TPS, relative TPS change, latency speedup and paired
recoveries/losses. `pairs.csv` stores per-example times, answer tokens, forwards
and scores. TPS is total visible answer tokens / total synchronized decode time,
excluding EOS. EOS/answer length may differ between policies: TPS is not a
fixed-work measurement or a reasoning-quality score.

`samples/00000/result.json` keeps final text/IDs, original and frozen secondary
scores, group token counts, actual forward counts, peak GPU allocation and compact
engine accounting. Pruning retains applied positions/event budgets; sparse keeps
anchors and actual attention pair counts. No hidden representations or attention
matrices are written. Reports update after each completed example. `.incomplete`
has `partial.json` after each policy but is excluded from aggregate reports.

`config.json` pins source hash, model commits, input hash, policies and decoder.
Resume skips completed examples and rejects changed source/config/inputs. Keep
the same sample count on resume; use a fresh run name to change it. Completed
old runs remain intact. The first two baseline visible outputs must match the
pinned official decoder; this verification is outside timing.

Read mechanisms in `pruning.py` and `sparse.py`; prefix-row reuse lives in
`../audio_drift/compact.py`. New runner: `run.py`. CPU correctness tests compare
sparse gathered attention with an independent dense-mask reference, plus full
pinned-transformer execution with cached query positions. Actual A40 speed and
full-model output quality must be measured on Vast.
