# Audio representation drift in DIFFA

Measures hidden-state and actual attention K/V changes while DIFFA denoises a
text response to audio. Run model inference on Vast; analyze results locally.
The default checkpoint is **DIFFA-2**, with its original semantic/acoustic
connectors, Whisper-large-v3, and LLaDA-8B-Instruct. This experiment does not
change the existing neighborhood, sleep, or translation experiments.

## Full-depth measurement correction

The initial GSM v7/v8 commands selected layers 0,7,15,23,31 to reduce diagnostic
overhead, and the quoted averages pooled 7,15,23,31. That was insufficient for
an all-depth claim. Updated `gsm-pilot` and `gsm-sweep` now default to **all 32
block outputs**, including layer 0. Layer 0 is the first transformer block's
output, not the input embedding. Existing running jobs keep their original
arguments; finish them before pulling source changes. Their quality/latency
results remain useful and their drift is valid for the measured layers.

After the current sweep finishes, collect full-depth GSM drift without rerunning
cache policies or latency benchmarks:

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_gsm_depth10_v9 SAMPLES=10 SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh gsm-depth > audio_gsm_depth10_v9.log 2>&1 &
tail -f audio_gsm_depth10_v9.log
```

This reuses the same manifest and 512/32/256 decoding budget as v8. Every example
has one instrumented baseline; the first two also have official verification.
There are no policy decodes, dense branches, fixed-input replays or pairwise
vector histories. All-layer hooks increase diagnostic overhead, so runtime is
not inferred from ordinary decode latency. No GPU execution is performed locally.

`analysis/cosine/measurements.json` includes per-example and aggregate
`depth_summary` / `aggregate_depth_summary` fields. They explicitly report layer
coverage, include layer 0, average only consecutive-forward comparisons up to
predicted content completion, and weight examples equally. Per-layer means and
`aggregate_drift_by_layer.png` prevent the global mean from hiding depth effects.
The legacy `active_mean_cosine_distance` field remains for historical comparison;
`legacy_active_mean_layers` identifies its layer-0-excluding pool. Use the new
full-depth fields for all-32-layer claims. Missing cells fail rather than being
silently omitted. Existing ASR v6 data already contains all 32 layers and can be
reanalyzed offline; the unrecorded GSM layers require a new baseline run.

## Push audio refresh intervals beyond text (v8)

The next controlled sweep reuses the existing 10-row GSM8K Speech manifest and
its audio files. It is a development experiment, not fresh held-out validation.
All baseline and cache outputs are regenerated at the new capacity.

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env \
  RUN_NAME=audio_gsm_sweep10_v8 \
  SAMPLES=10 \
  SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda \
  VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh gsm-sweep \
  > audio_gsm_sweep10_v8.log 2>&1 &
tail -f audio_gsm_sweep10_v8.log
```

| Policy | Acoustic interval | Prompt interval | Semantic interval |
|---|---:|---:|---:|
| Identity control | 1 | 1 | 1 |
| Shared prefix | 8 | 8 | 8 |
| Shared prefix, more aggressive | 16 | 16 | 16 |
| Acoustic slower | 16 | 8 | 8 |
| Acoustic slower | 32 | 8 | 8 |
| Acoustic permanent reuse after first forward | 0 | 8 | 8 |
| All audio slower | 16 | 8 | 16 |

Zero means initialize on the first forward and never refresh that group. Every
answer row still updates. Compare the audio-specific schedules against both
shared-prefix controls at measured latency and task accuracy.

**Answer capacity:** `LENGTH=512`, `BLOCK_LENGTH=32`, `STEPS=256` by default,
versus 256/32/128 in v7. Both schedules commit two token positions per forward;
the new budget permits up to 16 blocks. Autonomous benchmark decoding stops when
EOS is revealed and preceding masks are filled. The instrumented baseline and
its official verification still run the full fixed budget. Larger capacity adds
masked rows to each forward and can change diffusion outputs; compare policies
against the new baseline, not directly against v7 scores/timings. Longer capacity
reduces budget exhaustion risk but does not guarantee EOS or correct answers.
Missing EOS and visible answer-token counts are recorded.

**Two performance measurements:** every example gets ordinary EOS-stopped
baseline/policy decoding, including the actual generated explanation and final
answer. The first two examples also get a separate `fixed_input_replay.json`:
all seven policies receive the *same* preloaded baseline token states, in order,
for `min(64, baseline EOS-stopped forwards)` forwards. Every timed pass starts
with an empty cache. One warmup and two timed repeats per policy are bracketed
by baseline timings. Replay calls the same full-vocabulary forward path but does
not commit predictions or use the policy's generated answer length. Input device
transfers, drift hooks and task scoring are outside replay timing. Replay is a
forward-path timing measurement, **not** an autonomous generation or accuracy
measurement; its predicted tokens are never used in the scored decodes.
The identity's final replay-step top-1 predictions must match baseline; the
small upstream CPU test also checks identical input ordering and immutability.

`analysis/reasoning_report.md` and `reasoning_summary.json` show both speedup
types separately. The CSV records visible answer lengths, forward counts,
strict/normalized score outcomes and replay timing pairs. Override
`REPLAY_SAMPLES=0` to disable replay, or `REPLAY_STEPS`/`REPLAY_REPEATS` to adjust
its budget. A single-example run should also set `REPLAY_SAMPLES=1`.

**Scorer correction:** `numeric-labels-v2` accepts the literal `Answer:` and
`<final numerical answer>` labels after an explicit final marker, then applies
the same exact numeric comparison. This handles the two formats seen in v7;
it does not extract arbitrary numbers from reasoning. `strict_prediction`,
`strict_correct`, and `format_normalized` preserve visibility into the old strict
rule. Old downloaded results and runs are not overwritten.

All 10 examples run seven cache policies with one warmup and two timed repeats.
Only the first two have the extra equal-input replay. No dense intervention
branches, pairwise vector collection or layer-specific refresh schedules are
added. GPU runtime for this larger-capacity preset is not yet measured; use the
first replay-free sample (example 3) for steady-state runtime estimates. This
sweep should establish whether longer audio reuse helps before adding depth
schedules. New run names are required after source/settings changes.

## Spoken GSM8K refresh pilot (v7)

Uses `mistralai/gsm8k_speech`, synthetic spoken GSM8K test questions with final
numeric references. Resolves and records the dataset commit. Downloads/caches
three test Parquet shards (about 978 MB total) and consumes them with local
`ParquetFile` readers, avoiding the previous partially consumed streaming scanner
shutdown problem. Audio headers determine eligibility; a seeded reservoir selects
10 clips of at most 30 seconds without consulting model predictions. Longer clips
are excluded, never truncated. Only selected audio payloads are retained in RAM.
The prepared manifest and WAV files are reusable across runs.

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env \
  RUN_NAME=audio_gsm10_v7 \
  SAMPLES=10 \
  SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda \
  VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh gsm-pilot \
  > audio_gsm10_v7.log 2>&1 &
tail -f audio_gsm10_v7.log
```

For a two-example capability/timing check, use `SAMPLES=2` and a fresh run name
such as `audio_gsm2_v7`. Preparation still selects the same reusable 10-row
manifest. Subsequent 10-example runs need a fresh name; settings are fingerprinted.
No new Python dependencies are needed after v6 setup. The fixed prompt asks the
model to solve the **spoken** problem, think step by step, and end with
`#### <number>`. No reference answer or text transcript is supplied to the model.
The decoding budget is 256 answer slots, 32-slot blocks, 128 forwards; adjust only
in a fresh run. Missing EOS is recorded as possible budget exhaustion.

| Compact cache policy | Acoustic interval | Fixed prompt interval | Semantic interval |
|---|---:|---:|---:|
| Identity control | 1 | 1 | 1 |
| Shared prefix | 4 | 4 | 4 |
| Shared prefix | 8 | 8 | 8 |
| Acoustic slower | 8 | 4 | 4 |
| Acoustic slower still | 16 | 4 | 4 |
| Acoustic and prompt slower | 8 | 8 | 4 |

The ordinary uncached baseline collects scalar drift at layers 0, 7, 15, 23, 31.
Policies run from the first decoding forward with real row skipping, one warmup
and two synchronized timed repeats each. Every answer row remains active. There
are **no dense continuation branches**, no pairwise tensor history, and no
operator profiles in this preset. Verification checks the first two uncached
samples when `VERIFY_SAMPLES=2`; refresh-every-forward must match baseline tokens.
All policies use the same safe EOS stopping rule. Forward counts are reported
because different reasoning lengths can influence total latency.

`numeric` scoring accepts the last explicit `####` or `Final answer:` marker
followed by one numeric scalar (or a response consisting solely of a number).
Thousands grouping, optional dollar signs and decimal trailing zeros are
normalized using exact decimal arithmetic. It never takes the last arbitrary
number from an unmarked reasoning paragraph. Missing/malformed answers count as
incorrect and are separately counted as parse failures. Explanations are saved
for audit but are not graded. Unsupported output formats can be inspected in the
per-sample texts before drawing task-capability conclusions.

Results:

- `summary.json`: baseline and compact-policy final-answer accuracy and speedup.
- `analysis/reasoning_report.md`: paired accuracy, correct-to-wrong and
  wrong-to-correct counts, parse failures and measured decode speedup.
- `analysis/reasoning_pairs.csv`: per-example predictions, latency, forward counts
  and missing-EOS flags.
- `analysis/reasoning_summary.json`: machine-readable paired report including
  missing-EOS totals.
- `samples/*/result.json` and `cache_benchmark.json`: complete texts, scores,
  schedules, repeated timings, selected positions and memory accounting.

Ten examples are an exploratory capability/schedule pilot. Inspect the uncached
accuracy, parse failures and output-budget flags first. A weak baseline prevents a
useful cache-preservation conclusion. Tune on these examples, then validate the
selected schedules on fresh examples. Compare the asymmetric schedules against
both shared-prefix controls at measured latency and accuracy; a drift ratio alone
does not determine a safe refresh interval. GPU timings for this task are not yet
measured; use the first completed sample's wall time to estimate the remaining run.

To export the completed folder and log:

```bash
python - <<'PYZIP'
from pathlib import Path
import zipfile
run = Path('runs/audio_gsm10_v7')
assert (run / 'summary.json').is_file(), 'Run has not produced a summary yet'
out = Path('exports/audio_gsm10_v7.zip')
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(run.rglob('*')):
        if f.is_file():
            z.write(f, f.relative_to(run.parent))
    log = Path('audio_gsm10_v7.log')
    if log.is_file():
        z.write(log)
print(out, out.stat().st_size)
PYZIP
```

## Acoustic versus semantic timing study (v2)

On the existing Vast machine after the original smoke run:

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_groups128_v2 SKIP_INSTALL=1 GROUP_STUDY=1 \
  LENGTH=128 BLOCK_LENGTH=32 STEPS=128 VERIFY_SAMPLES=2 REFERENCE_DEVICE=cuda \
  bash scripts/vast_audio_drift.sh smoke > audio_groups128_v2.log 2>&1 &
tail -f audio_groups128_v2.log
```

This reuses the same two clips and installed packages, increases ASR output
capacity, and checks **both** uncached baseline outputs against the official
decoder. A fresh name is required: v1 results remain unchanged.

**All drift measurements come from the ordinary uncached baseline.** It runs
once per sample. Every intervention then starts from an identical saved baseline
token state, and runs sequentially on the same GPU. Baseline anchor KVs are
reconstructed with one forward; full representations are not archived.

`GROUP_STUDY=1` runs this matrix at four baseline anchor fractions:

| Condition | Acoustic KVs | Semantic KVs |
|---|---|---|
| Baseline | Recompute | Recompute |
| Acoustic-only freeze/reuse | Reuse | Recompute |
| Semantic-only freeze/reuse | Recompute | Reuse |
| Both freeze/reuse | Reuse | Reuse |
| Mixed A8:S2 | Refresh every 8 forwards | Refresh every 2 forwards |
| Mixed A2:S8 | Refresh every 2 forwards | Refresh every 8 forwards |
| Remove | No attention access | No attention access |
| Text-only restart | Embeddings absent | Embeddings absent |

For each of the three freeze groups, test permanent freezing and refresh
intervals **1, 2, 4, 8**. Interval 1 is an identity control that must generate
exactly the baseline's final IDs; the run stops if it fails. With an anchor at
step `a`, interval 4 reuses at `a+1`, `a+2`, `a+3` and refreshes at `a+4`.
Intervals continue across block boundaries. Refreshes use the intervention's
**own current representations**, never a future baseline representation.
Mixed schedules use independent intervals for the two disjoint token groups.

There are **76 sequential continuations per clip**, totaling 6,023 continuation
forwards at the 128-forward setting, plus baseline, verification, and anchor
replay. This is substantially more work than the first smoke run. It prints
baseline progress every 8 forwards and prints each condition before/after it
runs. There are still only two examples: validate transcription quality before
expanding the sample count or drawing conclusions.

Controls: `REFRESH_INTERVALS="1 2 4 8"`, `MIXED_REFRESH="8:2 2:8"`,
`BRANCH_FRACTIONS="0 .25 .5 .75"`, `PROGRESS_EVERY=8`. The Python CLI also
supports a subset, e.g. `--interventions freeze --freeze-groups acoustic semantic
--refresh-intervals 2 4 8 --mixed-refresh 8:2 2:8`.

Each branch records its group schedule, every refresh's step/cache age, commit
events, first-forward KL/top-1 changes on all masks **and on unresolved lexical
baseline-answer positions**, final text/score, and exact-token agreement. The
lexical subset excludes special tokens and positions after the baseline EOS;
it is posthoc diagnostic information, never input to the decoding policy.
`summary.json` separates branches with meaningful baseline content still
unresolved and includes matched baseline WER/deltas.

Plots separate acoustic, semantic, both, and mixed schedules, and show both
all branches and branches with content remaining. Audio/prompt heatmaps share
color scales for the same task, reference, representation, and metric.

The launcher also computes `analysis/stability_relative_l2_median.json` and
`analysis/stability_relative_l2_p90.json`. These sweep 1%, 2%, and 5% drift
thresholds per layer/group/reference and require four consecutive forwards.
They distinguish the **first sustained dip** from a window that stays below
threshold through the end. Thresholds are exploratory, not quality guarantees
or an implemented adaptive cache policy. A stable late tail may just be EOS
padding after the meaningful answer has already finished.

You can compute other posthoc thresholds from the saved baseline metrics:

```bash
python -m experiments.audio_drift.stability --run runs/audio_groups128_v2 \
  --metric cosine_distance --statistic p90 --thresholds .0001 .001 .01 --window 4
```

After checking the two longer baselines, prepare a larger, speaker-diverse
sample from the actual LibriSpeech test-clean split (not the dummy dataset):

```bash
source .venv-audio/bin/activate
python -m experiments.audio_drift.prepare \
  --dataset openslr/librispeech_asr --config clean --split test \
  --reader local-parquet --shuffle-seed 1729 --max-per-speaker 1 \
  --samples 20 --out data/audio_drift/test_clean20.jsonl
```

The helper records the immutable dataset revision, speaker IDs, filters, and
shuffle settings. It selects clips at most 30 seconds long and fails if too few
qualify. This is a reproducible filtered pilot, not a full official benchmark.
Then use `MANIFEST=$PWD/data/audio_drift/test_clean20.jsonl SAMPLES=20` with
`GROUP_STUDY=1` and the `pilot` mode. Plan for the printed forward budget.

V2 does not change model weights, prompt format, unmasking, or WER normalization.
It remains a sensitivity study: current audio projections are still computed
before substitution; these timings do **not** establish an execution speedup.

## First run on Vast

From this repository, using Python 3.10–3.12 and a BF16 CUDA GPU:

```bash
RUN_NAME=audio_smoke_v1 bash scripts/vast_audio_drift.sh smoke
```

This creates **`.venv-audio`**, installs inference-only dependencies, fetches the
pinned official DIFFA source, downloads the public checkpoints, and measures
two real speech clips from the LibriSpeech dummy validation set. It uses 32
answer slots, two blocks of 16, and 32 denoising forwards. It compares the first
sample's instrumented output with the official unmodified decoder and fails
if they differ. It then writes heatmaps. The smoke dataset checks execution;
two ASR examples do not establish a research finding.

Start with one A100 40 GB or a genuine 45–48 GB single GPU, batch size 1,
64 GB host RAM, and 100–150 GB disk. These are starting recommendations, not
measured peak requirements. `device.json` records actual VRAM;
`result.json` records peak allocated CUDA memory during collection.

The default installation uses PyTorch/torchaudio 2.6.0 with CUDA 12.4. A modified
4090 should use the Ada-compatible stack. A 5090 requires a suitable Blackwell
stack; set matching `TORCH_VERSION`, `TORCHAUDIO_VERSION`, and `TORCH_INDEX_URL`
for that host. Do not install this experiment into the parent repo's GPU venv:
DIFFA needs Transformers 4.49.0, while the parent experiment pins 4.46.3.

For a more complete smoke test of the sensitivity interventions:

```bash
RUN_NAME=audio_branches_smoke_v1 SKIP_INSTALL=1 INTERVENTIONS=1 \
  bash scripts/vast_audio_drift.sh smoke
```

Each branch continues decoding, so enabling all interventions substantially
increases inference cost. First validate ordinary collection.

## Task manifest

Use a UTF-8 JSONL file (a JSON list also works). Each row needs a unique `id`,
an `audio` path relative to the manifest, a `task` label, and the complete text
`prompt`. Audio must be nonempty and at most 30 seconds. Longer clips fail
explicitly; segment them before running. Stereo is averaged to mono and audio
is resampled to 16 kHz. There is no silent audio or prompt truncation.

Example schema; replace the paths/prompts/targets with actual benchmark items:

```json
{"id":"clip1_asr","audio":"clips/clip1.wav","task":"asr","prompt":"Transcribe the speech. Return only the transcription.","target":"the actual reference transcript","metric":"wer"}
{"id":"clip1_qa","audio":"clips/clip1.wav","task":"semantic_qa","prompt":"Where is the speaker going?\nA. Home\nB. School\nC. Work\nAnswer with one letter.","target":"B","metric":"mcq"}
{"id":"clip1_emotion","audio":"clips/clip1.wav","task":"prosody_qa","prompt":"What emotion does the speaker convey?\nA. Anger\nB. Joy\nC. Sadness\nAnswer with one letter.","target":"A","metric":"mcq"}
```

Scoring options: `wer` (case/punctuation-insensitive word edit distance), `mcq`
(a conservative leading answer letter), `exact` (normalized word sequence),
or `none` (no target required). Include choices and answer-format instructions
directly in `prompt`; nothing is injected beyond the official DIFFA chat format.
Reference targets are used only after generation for scoring.

```bash
MANIFEST=/workspace/data/audio_tasks.jsonl RUN_NAME=audio_pilot_b32_v1 \
  SAMPLES=100 SKIP_INSTALL=1 bash scripts/vast_audio_drift.sh pilot
```

The pilot takes the first N manifest rows, **not N per task**. Supply 300 rows
and set `SAMPLES=300` for 100 examples of each of three tasks. The manifest
controls order and selection, and may reuse the same clip across questions.

Compare blocks with fresh output directories, holding answer length and
denoising budget constant:

```bash
MANIFEST=/workspace/data/audio_tasks.jsonl RUN_NAME=audio_b128_v1 \
  LENGTH=128 STEPS=128 BLOCK_LENGTH=128 SAMPLES=300 SKIP_INSTALL=1 \
  bash scripts/vast_audio_drift.sh pilot
```

`STEPS` must be no larger than `LENGTH` and divide evenly across blocks.
Decoding is deterministic confidence-based fixed-budget unmasking, matching
`origin_generate.py`: no sampling, CFG, KV reuse, factor decoding, or early EOS
stop. Every answer slot is filled. Display/scoring ends at the first EOS.

## Measurements and controls

For every selected layer and every forward:

* **Hidden state:** transformer block output, before the final model-wide norm.
* **Key/value:** inputs to the block's actual attention operation, after
  key/query normalization and RoPE, before optional grouped-head expansion.
* **Reference points:** previous forward, initial forward, and the first
  forward of the current block. Initial and block references remain fixed.
* **Metrics:** relative L2, cosine similarity/distance, and zero-reference-norm
  frequency. Relative L2 divides by the reference norm (epsilon `1e-8`).
* **Groups:** semantic audio, acoustic audio, combined audio, fixed text
  prompt, unchanged masked answer tokens, unchanged decoded answer tokens,
  and answer positions whose token identity changed since the reference.
* **Summaries:** count, mean, median, p90, maximum, computed across tokens in
  each group. Heads/features are flattened within each token, not pooled
  across positions. Prompt controls include chat headers and special tokens.
* **Metadata:** task, duration, sequence lengths, global and active-block mask
  fractions, block boundaries, and number/positions of newly committed tokens.

Stable decoded/masked controls require the same token identity in both
forwards. Newly revealed tokens are reported separately, so changed input
embeddings do not masquerade as contextual drift. Population membership can
still vary over time; scalar per-token files allow later cohort comparisons.

The first-layer audio/prompt KVs should be invariant: they are projections of
fixed input embeddings with fixed positions. That is a correctness control,
not evidence of an audio-specific discovery. Later-layer KVs can depend on
the developing answer. Hidden outputs of the first layer can also change.

Reference tensors retain their native dtype; differences are computed in
FP32. Reference banks live on CPU by default and have constant size in the
number of denoising steps. This incurs transfer/measurement overhead. Use
`REFERENCE_DEVICE=cuda` if there is enough VRAM. **Timing is instrumented and
must not be interpreted as ordinary decoding speed.**

## Optional causal interventions

`INTERVENTIONS=1` runs all three; the Python CLI permits a subset:

```bash
source .venv-audio/bin/activate
python -m experiments.audio_drift.run \
  --manifest /workspace/data/audio_tasks.jsonl --out runs/audio_freeze_v1 \
  --length 128 --block-length 32 --steps 128 \
  --interventions freeze remove --branch-fractions 0 .25 .5 .75
```

Anchors are selected by fraction of the baseline's forward calls, rounded to
the nearest step; duplicates are removed. Each intervention starts on the
**next forward after its anchor**, from exactly the baseline's saved token IDs:

* `freeze`: capture audio KVs at the anchor and substitute them at all layers
  during continuation. Queries and other KVs continue to update.
* `remove`: exclude all audio K/V columns for **every query at every layer**.
  Sequence positions and RoPE remain unchanged. This blocks both direct and
  indirect reads through recomputed audio positions.
* `text_only`: remove audio embeddings entirely and recompute from token IDs.
  Positions after audio shift. Treat this as an auxiliary text-only restart,
  not an isolated measure of the effect of audio access.

The first changed forward is compared against ordinary inference at the exact
same token state, using KL(baseline || intervention) and top-1 disagreement on
all still-masked answer slots, including future blocks. Final continuations
use the same remaining schedule and are scored against the same target.

`baseline_content_masks_remaining` counts unresolved positions before the
baseline's first EOS (excluding special tokens in v2). This is a posthoc
diagnostic, never a decoding signal.
If `baseline_answer_already_revealed` is true, unchanged final accuracy can be
trivial: the meaningful answer was already committed before the intervention.
Distinguish those branches when interpreting continued audio dependence.

There are **no persistent text hidden states in the uncached decoder**. The
`remove` test recomputes text states while preserving their positions; any
retained audio information must come through already revealed text IDs.

Frozen KVs are substituted after computing the current projections. These are
**sensitivity experiments, not optimized cache kernels**. Their diagnostic
elapsed times do not establish a speedup. To establish one, implement execution
that actually skips audio computation after identifying a useful policy.

## Results, plots, and resume

`runs/NAME/` contains:

* `config.json`: settings, source fingerprint, upstream/checkpoint revisions,
  selected inputs, and SHA256 of each audio file.
* `load_report.json`: adapter/LoRA loading details. LoRA alpha defaults to 16,
  matching upstream inference; missing adapter or LoRA weights fail loudly.
* `samples/00000/drift.jsonl`: streaming group summaries by layer, step,
  representation, and reference.
* `samples/00000/token_metrics/STEP.npz`: per-position relative L2 and cosine
  similarity only, with sequence IDs and group labels; **no representation
  vectors**. Opt in with `TOKEN_METRICS=1` (Python CLI: `--token-metrics`).
  Load with `np.load(path, allow_pickle=False)`.
* `steps.json`, `result.json`: schedule events, baseline text/score, upstream
  verification status, memory, and optional intervention results.
* `summary.json`: task-level accuracy and corpus WER by condition.
* `analysis/`: heatmaps, intervention curves, and numeric heatmap values.

Heatmaps show the mean across samples of token medians, first averaging steps
within each progress bin. They are **not pooled medians across samples**.
Empty bins stay missing. Age-zero comparisons are omitted from plots.
Intervention WER curves show mean per-sample WER; `summary.json` reports corpus
WER. V2 shares color scales across acoustic/semantic/prompt panels for matching
task, reference, representation, and metric; decoded-answer panels use their own scale.

Run analysis again without any GPU/model downloads:

```bash
python -m experiments.audio_drift.analyze --run runs/audio_pilot_b32_v1
```

Repeat the same command with the same `RUN_NAME` to resume: completed samples
are skipped, an interrupted sample restarts. Config/source/input changes are
rejected; use a fresh name. Outputs are committed per sample after successful
completion. Download the results before terminating the Vast instance.

Group summaries are the default. `TOKEN_METRICS=1` additionally retains
per-position scalar measurements; these can still reach hundreds of MB per
example for long runs over all layers. Full activation snapshots are
explicitly opt-in, using `--snapshot-steps 0 16 --snapshot-layers 0 15 31` in
the Python CLI. They can be large. Layer selection: `LAYERS=0,7,15,23,31`.

## Duration-stratified LibriSpeech pilot and baseline profiling

Use `long-pilot` for the next study rather than repeating the 76-branch smoke
matrix. It prepares **20 distinct clips from LibriSpeech clean/test**, with
10 clips in `[10, 20)` seconds and 10 in `[20, 30]` seconds. Seeded selection
limits each speaker to at most two clips across both bins, hence at
least ten speakers. Bins are interleaved in the manifest, so `SAMPLES=10` also
gives a balanced first subset. Selection is duration-based, not quality-based;
the immutable dataset revision, durations, speaker IDs and quotas are saved.
It fails explicitly if the source cannot fill the quotas.

V4 uses `--reader local-parquet`: it downloads/caches **only** the selected
`clean/test/*.parquet` shards (not training data), reads the split fully using
`ParquetFile.iter_batches(use_threads=False)`, and closes all files before
selection. This avoids the partially consumed HF/Arrow streaming-scanner
shutdown hang/abort reported in https://github.com/apache/arrow/issues/45214.
It does not change installed dependencies or suppress preparation failures.
V4 shuffles the local split with Python's seeded RNG, so its selected IDs may
change relative to V3's HF streaming shuffle. The new manifest name preserves
V3 inputs. Progress logs show shard reads, scanned rows, and every selected clip.

Intervention KVs now default to `CACHE_DEVICE=model`: snapshots and refreshes
remain on the model's GPU. `REFERENCE_DEVICE` controls drift references
separately. Group banks and anchor banks own independent storage; CUDA storage
avoids the old per-layer GPU/CPU round trips and CPU snapshot cloning. This
changes diagnostic overhead, not the caching policy or the fact that fresh
representations are still computed. `CACHE_DEVICE=cpu` remains available.
`CPU_THREADS=4` limits small CPU diagnostic operations to four threads by default.
Branch and complete-sample elapsed times are printed in the log.

First validate two clips (one per duration range) before a longer run:

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_libri2_v4 SAMPLES=2 SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh long-pilot \
  > audio_libri2_v4.log 2>&1 &
tail -f audio_libri2_v4.log
```

This prepares the full 20-clip manifest but runs only its first two clips.
Use the actual timings to estimate the rest; no new duration is promised before
that GPU check. The next command runs all 20 under a separate run name.

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_libri20_v4 SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh long-pilot \
  > audio_libri20_v4.log 2>&1 &
tail -f audio_libri20_v4.log
```

Use a fresh run name when changing sample count or code. To run only the first
ten examples, add `SAMPLES=10` and use `RUN_NAME=audio_libri10_v4`. No reinstall
or model weight download is needed if the previous audio environment/cache
still exists. Creating the new manifest does download real test-set audio.

Defaults are a 256-slot answer, 32-slot blocks, and 128 forwards (two commits
per forward). The larger answer budget accommodates longer transcriptions;
the baseline is still checked against the official decoder with identical
settings. Fixed-budget decoding does not stop at EOS. Results record first
EOS position, missing EOS, and the step at which baseline content was revealed
so padding work and possible answer truncation remain visible.

Eight continuations per clip all start immediately after the first forward:

| Condition | Acoustic KV interval | Semantic KV interval |
|---|---:|---:|
| `freeze_acoustic` | Never refresh | Every forward |
| `freeze_semantic` | Every forward | Never refresh |
| `freeze` | Never refresh | Never refresh |
| `freeze_both_r1` | 1 | 1 |
| `freeze_both_r4` | 4 | 4 |
| `freeze_both_r8` | 8 | 8 |
| `freeze_mixed_a8_s2` | 8 | 2 |
| `freeze_mixed_a2_s8` | 2 | 8 |

This is 160 branches over 20 clips, rather than 1,520. No later anchors or
audio-removal branches are included. `--condition-names` selects explicit
names from the generated matrix and rejects typos. Baseline drift is collected
at layers `0,7,15,23,31` by default; `LAYERS=all` opts into full-depth metrics.
The intervention still acts on **every** transformer layer. Scalar group
summaries are saved; no hidden-state vectors or per-position metrics are saved
by default.

The first two examples (one per duration bin) also get a **separate baseline
profile** (`PROFILE_SAMPLES=0` disables it):

* `samples/00000/baseline_profile.json`: two warmed, synchronized wall-time
  repeats each of audio preparation and complete decoding. Decoding has no
  drift hooks, no profiler scopes and no CPU activation copies; it still uses
  the ordinary decoder's bookkeeping. Every repeated output must match baseline.
* `baseline_operators.txt`: major CPU/CUDA operators from sampled forwards.
* `baseline_trace.json`: Chrome/Perfetto trace for forward 0, the middle of the
  first block, and the final forward. Scopes label QKV projections, attention,
  feed-forward projections, attention output projection, and vocabulary output.

Audio preparation includes CPU feature extraction, transfers, speech encoder,
adapters and prompt setup; it excludes model loading and disk audio loading.
It happens once per request, not every denoising forward. Timing repeats use
already-resident audio. Traces are separate from clean timing repeats and have
profiling overhead. Projection and attention times cover **all sequence tokens**;
they do not directly identify audio-only runtime. No optimized cache execution
or speedup is claimed. Dense token fractions are not latency fractions, and
text queries must still attend to cached audio KVs.

`baseline.json` and `steps.json` are saved before profiles/branches; each
completed branch is flushed to `interventions.jsonl` immediately. During an
unfinished sample these live in `samples/.00000.incomplete/`. Resume still
restarts an incomplete sample; partial files are for inspection, not branch
resume checkpoints.

Package the completed run, including hidden partial directories if applicable:

```bash
python - <<'PY'
from pathlib import Path
import zipfile
root = Path('/workspace/llada-audio-drift')
run = root / 'runs/audio_libri20_v4'
assert (run / 'summary.json').is_file(), 'No completed-sample summary found'
out = Path('/workspace/audio_libri20_v4.zip')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in run.rglob('*'):
        if p.is_file():
            z.write(p, p.relative_to(root))
    for p in (root / 'audio_libri20_v4.log',
              root / 'data/audio_drift/librispeech_duration20_v4.provenance.json'):
        if p.is_file():
            z.write(p, p.relative_to(root))
print(out, out.stat().st_size)
PY
```

## Prompt controls and actual row-skipping benchmark (v5)

Three questions are tested separately:

1. **Audio versus fixed text prompt:** baseline hidden/K/V drift already records
   prompt tokens on shared plot scales. V5 additionally freezes prompt KVs
   permanently, refreshes them every four forwards, and includes a prompt
   interval-one identity control. `prompt` means every fixed text token, before
   and after the inserted audio; it never includes generated answer tokens.
2. **Within audio:** retain the acoustic/semantic permanent and mixed schedules.
3. **Actual acceleration:** time an adapter that skips cached prefix rows, not
   the diagnostic KV substitution hooks. No answer row is skipped, masked or
   decoded. Text queries can still attend to every cached audio/prompt KV.

Start with the same two longer clips and reuse the existing V4 manifest:

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_controls2_v5 SAMPLES=2 SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh control-pilot \
  > audio_controls2_v5.log 2>&1 &
tail -f audio_controls2_v5.log
```

There are **13 diagnostic branches per sample**, all anchored at zero: the
previous eight conditions plus `freeze_prompt`, `freeze_prompt_r1`,
`freeze_prompt_r4`, `freeze_audio_matched_r4`, and `freeze_prefix_r4`.
`audio_matched` selects a fixed, evenly spaced audio subset of the same size
as the text prompt (or all audio if there are fewer audio tokens; inspect saved
positions). This controls token count when comparing prompt and audio caching.
It is an exploratory fixed subset, not a learned token importance method.
`prefix` selects all fixed text and audio, providing a generic prefix cache
comparison. Individual selected positions are saved in diagnostic results.

After the dense sensitivity runs, the first two samples benchmark **seven**
conditions: prompt-r4, matched-audio-r4, both-audio-r4, whole-prefix-r4,
permanent acoustic reuse, mixed A8:S2, and both-r1 identity.
`BENCHMARK_SAMPLES=0` disables these timings. `BENCHMARK_REPEATS=2` is the pilot
setting; increase repeats on a later validation run. Default control-pilot
sample count is two. No new full 20-sample run starts automatically.

The adapter in `compact.py` is restricted to the pinned eval-mode, ungrouped
LLaDALlamaBlock. It leaves the original upstream source untouched. On the first
forward and complete refreshes it calls upstream's block implementation; on
reuse forwards it gathers active rows, applies QKV, absolute-position RoPE,
attention queries/output, and MLP only to those rows, updates their KV slots,
and retains cached rows' prior block outputs. Keys/values still span the whole
sequence. Mixed schedules refresh each group's own branch states independently.
The full embedding, final normalization and vocabulary head are left identical
in baseline and cached runs; they are not optimized specially for cached runs.

**This changes execution, not the intended KV reuse policy.** CPU tests compare
per-forward answer logits to dense KV substitution for noncontiguous prompt
positions, audio, whole prefix, matched audio, permanent/periodic and mixed
schedules. Hooks verify projection row counts decrease. CUDA GEMM shape changes
can introduce numerical differences; timed results record whether visible
answer IDs match the paired dense intervention as well as baseline. Interval
one must match baseline visible tokens exactly. CUDA quality/speed remain to
be measured; positive speedup is not assumed.

`cache_benchmark.json` is flushed after each timed policy. It records:

* One untimed warmup per baseline/policy; two synchronized wall-time repeats per
  policy. Each repeat starts generation and its cache afresh, including initial
  cache population, refresh work, gathers/scatters and decoder bookkeeping.
  Task scoring is outside the timer. There are no drift or profiling hooks.
* Warmed baseline timings before and after the policy sweep, their median, and
  `decode_speedup = baseline_median / policy_median`. Values below one are
  slowdowns and are retained. Individual repeats expose noise/clock drift.
* WER, text, visible token IDs, forward counts, selected positions, peak GPU
  allocation, and computed/skipped token-layer updates for every repeat.
  Skipped-update fractions are **not** total FLOP or latency savings.
* The sample's separately synchronized audio preparation time. The ratio with
  that shared one-time cost added to both decode medians is saved separately;
  model loading and disk audio I/O remain excluded. Preparation itself is not
  accelerated by prefix caching.

Timing defaults to **safe EOS stopping**: stop only after a revealed EOS has
no unresolved answer positions before it. Fixed-budget diagnostic collection
still runs all 128 forwards and is verified against upstream. The clean EOS
baseline must have the same visible answer IDs as the fixed-budget official
baseline. Baseline and cached timing runs use identical stopping and answer
capacity. Different policies may finish at different forward counts; those
counts are reported so fewer steps are not mistaken for cheaper forwards.
Use `BENCHMARK_STOPPING=fixed` to benchmark all 128 forwards instead, under a
fresh run name. Both settings retain all answer query rows during each forward.

`summary.json` keeps dense intervention quality under `tasks` and aggregates
actual timings/quality separately under `cache_benchmarks`, using the ratio of
summed paired baseline and policy median times. The performance subset may be
smaller than the diagnostic subset. The ordinary baseline profile previously
showed mostly feed-forward/projection work, so skipping those token rows is
necessary to test practical savings rather than only replacing their KVs.

## Cosine drift and step-pair similarity figures (v6)

`cosine-pilot` collects **baseline only** on the same two clips. It does not
rerun caching policies or benchmarks. Scalar hidden/K/V drift is recorded at
all 32 block-output layers. Additionally, it computes full step-pair hidden
cosine matrices for the first 64 forwards at layers 7,15,23,31, separately for
acoustic, semantic and fixed prompt tokens:

```bash
cd /workspace/llada-audio-drift
git pull --ff-only
nohup env RUN_NAME=audio_cosine2_v6 SAMPLES=2 SKIP_INSTALL=1 \
  REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_drift.sh cosine-pilot \
  > audio_cosine2_v6.log 2>&1 &
tail -f audio_cosine2_v6.log
```

This reproduces the *type* of VidLaDA Figures 4–5, not a claim that our dataset,
model, token pooling and layer aggregation are identical to theirs. Figure 4's
exact pooling/aggregation protocol is not established from its caption.
Our explicit definition is **mean cosine at corresponding token positions**,
then equal weighting of selected layers for aggregate plots. Cosine of a mean
embedding would be a different metric and is not substituted. Averages of
per-token medians in earlier relative-L2 summaries are also a different
statistic: do not compare them numerically to these mean cosine values.

`PAIRWISE_COUNT=64`, `PAIRWISE_STRIDE=1`, `PAIRWISE_LAYERS=7,15,23,31`, and
`PAIRWISE_DEVICE=cuda` are configurable. With 128 decoding forwards the default
pairwise matrices cover forward indices 0–63, not 64 evenly spaced points.
`PAIRWISE_COUNT=128` covers all pairs over all 128 forwards. Pairwise layers must
be included in the observed `LAYERS` selection. `PAIRWISE_LAYERS=all` retains
all observed layers and substantially increases memory. For the current
28-second clip, four layers and 64 snapshots require about **1.95 GB extra**
on the chosen device; full 32-layer pairwise retention is about eight times
that. The memory estimate is printed before collection.

Only selected fixed-prefix hidden vectors are normalized and retained **in
memory**, in FP32. Matmul TF32 is disabled by the runner. Previously observed
normalized vectors are dotted with each new snapshot, yielding exact token-mean
cosine for our definition, subject to FP32 rounding. Both-zero tokens have
similarity one. Vectors are released once the last selected step is processed.
They are never written to disk. `pairwise_cosine.jsonl` is flushed every forward
for partial inspection; `pairwise_cosine.json` holds scalar symmetric matrices,
selected steps/layers and a completeness flag. No answer representations enter
these matrices. No similarity between unmeasured steps is inferred from existing
scalar summaries.

The offline command also works on existing V4/V5 downloads:

```bash
python -m experiments.audio_drift.paper_plots --run runs/audio_controls2_v5
```

Existing runs can produce mean cosine-distance layer heatmaps and similarity
curves against forward zero; they cannot recover missing all-pairs values.
New runs additionally produce three-group similarity matrices on a shared
color scale, plus fixed `[0.8, 1]` versions matching the published plot range.
Individual layer matrices and numeric aggregate values remain available.
Output is `analysis/cosine/`. Both full and active-answer-cropped figures are
written. Completion refers to the **baseline's predicted** content, so truncated
baselines remain visible as quality failures. Post-answer regions are marked;
small drift after answer completion does not establish early cacheability.
Layer zero is our first transformer block output, not the input embedding.

## Validation

Without optional GPU packages, pure manifest/scoring tests run and tensor
tests skip. In the audio environment:

```bash
python -m unittest discover -s tests -p test_audio_drift.py -v
DIFFA_ROOT=$PWD/external/DIFFA \
  python -m unittest discover -s tests -p test_audio_drift.py -v
```

The second command exercises a tiny **real upstream LLaDA transformer on CPU**:
instrumented/official decoding equality, complete hooks, freeze identity at
the capture state, removal effects, and hook restoration after exceptions.
It downloads no model weights. Actual 8B checkpoint loading, CUDA peak memory,
and task quality must still be validated by the Vast smoke run.

Official source is pinned to `0a87594e110984fbaf00ebf5bfe5840c212e4438` and is
never edited or silently updated. DIFFA-2 and LLaDA weights have pinned default
revisions; Whisper's resolved revision is recorded. Existing checkouts with
changes are rejected. The loader follows the official encoder replacement
and LoRA merge, while avoiding its hard-coded local checkpoint paths.
