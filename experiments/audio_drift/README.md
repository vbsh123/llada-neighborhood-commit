# Audio representation drift in DIFFA

Measures hidden-state and actual attention K/V changes while DIFFA denoises a
text response to audio. Run model inference on Vast; analyze results locally.
The default checkpoint is **DIFFA-2**, with its original semantic/acoustic
connectors, Whisper-large-v3, and LLaDA-8B-Instruct. This experiment does not
change the existing neighborhood, sleep, or translation experiments.

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
  --streaming --shuffle-seed 1729 --shuffle-buffer 1000 --max-per-speaker 1 \
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
