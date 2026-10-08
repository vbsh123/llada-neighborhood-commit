# LLaDA neighborhood commitment experiment

GSM8K experiment: ordinary highest-confidence top1 and a top1-anchored
three- or four-position window. Independent repository; no model inference runs locally.
Collector derived from [dllm-confidence-geography](https://github.com/vbsh123/dllm-confidence-geography).

## Decoder

`top1_window` finds the highest-confidence eligible masked position p, then
commits p and masked eligible positions p−1, p+1. All predictions come
from the same forward. The anchor is always included, so the decoder progresses.
No neighbor confidence gate. Filled slots and positions outside the active
response block are skipped, without replacing them with other positions.
Pass `--window-size 4` to also commit p+2, restoring the original four-token rule.
The default remains three tokens; the choice is saved in the run configuration.

Selection is in `confidence_geography/core.py:select()`. The decoding loop and
trace collection are in `confidence_geography/run.py:collect_sample()`.
Read [QUIRKS.md](QUIRKS.md) for the explicit protocol choices.

Version 2 removes +2: at most three tokens per forward. Earlier four-position
results belong to version 1. Use a fresh output directory after updating source.

## Vast commands

Use a CUDA image with Python 3.10–3.12 and BF16-capable GPU. The 8B model plus
full-sequence logits needs sufficient VRAM; a 48 GB GPU is the intended pilot
starting point, not a tested minimum for this new experiment.

```bash
git clone https://github.com/vbsh123/llada-neighborhood-commit.git
cd llada-neighborhood-commit
RUN_NAME=window_smoke_v1 bash scripts/vast.sh smoke
```

The smoke run installs dependencies, downloads the checkpoint, and actually
runs both policies on two questions with 64 response slots. It requires a GPU.

After smoke succeeds, run the paired 100-question, 256-slot pilot:

```bash
RUN_NAME=window_pilot_v1 SKIP_INSTALL=1 bash scripts/vast.sh pilot
```

Or in the background:

```bash
nohup env RUN_NAME=window_pilot_v1 SKIP_INSTALL=1 bash scripts/vast.sh pilot > window_pilot_v1.log 2>&1 &
tail -f window_pilot_v1.log
```

Use a fresh name after changing code or settings. To resume an interrupted run,
use its same name, unchanged source, and identical settings. Completed questions
are skipped; an interrupted question restarts. Seeds, questions, model revision,
dataset commit, prompt, and scoring match across the two policies.

Knobs: `SAMPLES`, `LENGTH`, `BLOCK_LENGTH` (default 0 = full response), `SEED`
(1729), `OFFSET` (0), `DATASET_REVISION` (resolved and recorded on first launch),
and `SKIP_INSTALL=1` after successful setup. The wrapper runs top1 first.

## Results and export

`runs/window_pilot_v1/window_comparison.json` reports numeric accuracy, paired
correctness changes, forwards saved, instrumented runtime, actual batch sizes,
and confidence of forced neighbors. Full per-position confidence, entropy,
top-k, states, and exact window selections are stored in each policy's traces.

The script exports the run to `exports/*.tar.gz` with a SHA256 checksum. Download
that archive before closing the machine; it contains no checkpoint or HF auth.

For offline comparison after downloading:

```bash
python -m confidence_geography.compare_window \
  --baseline runs/window_pilot_v1/top1 \
  --window runs/window_pilot_v1/top1_window \
  --out runs/window_pilot_v1/window_comparison.json
```

CPU checks (no packages, model, or GPU needed):

```bash
python -m unittest discover -s tests -v
```

This experiment tests the locality hypothesis; it does not assume simultaneous
neighbor commitment preserves accuracy. It does not reproduce another method.

## One-forward repair of a completed window run

This reuses saved outputs and traces. It does **not** rerun generation or top1.
It selects forced neighbors whose original commitment confidence was strictly
below 0.75, excluding anchors, originally special tokens, and positions at or
after the original final stop. All selected positions are masked together and
filled from one model forward, without a confidence acceptance gate or retries.
Questions with no candidates require no forward. Both the earlier four-token
window and the current three-token window are supported.

On the Vast machine with your completed run and existing environment:

```bash
git pull
source .venv/bin/activate
python -m confidence_geography.repair_window \
  --run runs/window_3tokens_v1/top1_window \
  --out runs/window_3tokens_repair75_v1 \
  --threshold 0.75
bash scripts/export.sh runs/window_3tokens_repair75_v1
```

Use the same command to resume; completed repairs are skipped. Use a fresh
output folder when changing settings or code. The model/tokenizer revision and
prompt IDs come from the saved run. It requires the same GPU dependencies as
generation; there is no extra installation step when the environment is ready.

`summary.json` reports original versus repaired accuracy, paired correctness
changes, candidate counts, changed tokens, and extra forwards. Each sample has
its unchanged `original_result.json` and separate repaired `result.json`, with
candidate metadata and new predictions. These outputs are results, not new
generation traces. The source run is untouched.

Read `confidence_geography/repair_window.py`: `repair_candidates()` selects the
positions, `repair_window()` performs the single forward, and `repaired_result()`
decodes and scores the repaired answer. CPU tests use a fake model; actual repair
quality must be measured on Vast.

## Four-token generation followed by repair on a new machine

After installing the GPU environment, run only the four-token window and its
repair (no separate top1 baseline):

```bash
source .venv/bin/activate
python -m confidence_geography.run \
  --out runs/window4_repair75_v1/top1_window \
  --policy top1_window --window-size 4 \
  --samples 100 --length 256 --seed 1729
python -m confidence_geography.repair_window \
  --run runs/window4_repair75_v1/top1_window \
  --out runs/window4_repair75_v1/repaired --threshold 0.75
bash scripts/export.sh runs/window4_repair75_v1
```

The repair runs after the generation command completes and uses its exact saved
outputs. The original and repaired answers are both retained. The original
answer's generation is ungated; the repair threshold only selects positions
for the final simultaneous repair forward.

## Add a matched top1 baseline after window and repair finish

No window or repair rerun is needed. This command loads the saved sample list
directly, preserving its order, dataset indices and IDs without shuffling or
downloading GSM8K again. It copies the source configuration and changes only
`policy` to `top1`, including the original model revision and response length.
The unused `window_size` setting is retained for exact configuration matching.
The source window must be complete. It is only read, never resumed or modified.

```bash
git pull
source .venv/bin/activate
python -u -m confidence_geography.matched_baseline \
  --window runs/window4_repair75_500_v1/top1_window \
  --out runs/window4_repair75_500_v1/top1
python -u -m confidence_geography.compare_window \
  --baseline runs/window4_repair75_500_v1/top1 \
  --window runs/window4_repair75_500_v1/top1_window \
  --repaired runs/window4_repair75_500_v1/repaired \
  --out runs/window4_repair75_500_v1/three_way_comparison.json
bash scripts/export.sh runs/window4_repair75_500_v1
```

The optional `--repaired` comparison checks that each repair used the exact
original window result. It reports accuracy and paired correctness, plus
generation and repair forwards/time added together. Timing is instrumented
work, not total command wall time; checkpoint loading and trace reading are
excluded. No equal-hardware assumption is enforced: compare GPU environments
in run manifests before interpreting timing differences.

Source updates prevent resuming older generation/repair runs under changed code;
they do not prevent reading their completed results for this new baseline and
comparison. Use a fresh baseline output folder if that folder already contains
an incompatible run. Code: `matched_baseline.py:matched_inputs()` validates and
copies the inputs; `run_matched_baseline()` calls only the existing top1 decoder.

## SOAR discarded-branch measurement

A separate observation-only probe is in
[experiments/soar_reuse](experiments/soar_reuse/README.md). It runs pinned official
SOAR on saved prompts and logs discarded token opportunities, branch conflicts,
survivor support, entropy and cross-branch disagreement. It makes no merges and
does not establish a merge speedup. Start with its two-question Vast smoke;
only CPU logic checks have run locally. Existing window/repair outputs are read,
not regenerated. Probe code lives outside the collector package, so adding it
does not change existing generation/repair source fingerprints.
