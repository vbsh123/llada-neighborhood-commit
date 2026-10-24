# DIFFA prefix cache experiment

Standalone **orphan branch `audio-cache-clean`**: no inherited experiment tree or
imports from the older repo. Only fixed-prefix caching is implemented here.
Prior experiments and results remain on `audio-drift-study`.

Default experiment: **only16/16/16 on the exact previous50 questions**.
`audio_cache/cohort.json` records their source rows, prompts, references and WAV
hashes from the downloaded `audio_independent_combined50_v22.zip`.
Preparation restores those rows directly from the pinned dataset and verifies
every original audio hash and the complete input fingerprint. It does not select
another random set or require the previous machine's files.

```bash
cd /workspace
git clone --single-branch --branch audio-cache-clean \
  https://github.com/vbsh123/llada-neighborhood-commit.git diffa-cache-clean
cd diffa-cache-clean
nohup bash scripts/run.sh > cache161616_only50_v29.log 2>&1 &
tail -f cache161616_only50_v29.log
```

The script creates its environment, installs pinned dependencies, checks out the
unmodified pinned DIFFA source, restores the50 inputs, then evaluates. A BF16 CUDA
GPU is required; an A40 is suitable. Run only one job per GPU. Closing `tail` with
Ctrl-C does not stop the experiment. For an unchanged interrupted run, use the
same command with `SKIP_INSTALL=1`; completed examples are skipped.

The only generated condition is **16/16/16**. One process warmup of that policy;
one timed EOS decode per example. The saved baseline outputs from the original
archive are rescored with the frozen extractor (26/50 correct). There are **no
uncached baseline warmups, timed baseline decodes or official verification
decodes** in this mode. No drift probes,
pruning, sparse attention, boosting, replay or per-example repeated warmups.

Decoder:512 answer slots,32-position commitment blocks,256 maximum forwards,
fixed quota of2 commits per forward. Model attention remains bidirectional over
the complete canvas. Every answer row is updated each forward. Caching skips the
fixed-prefix QKV/query/MLP rows, reusing post-RoPE K/V and block outputs; all KV
columns remain accessible. Original token positions are preserved.

Results: `runs/cache161616_only50_v29/analysis/report.md` and `pairs.csv`, with original
and frozen secondary scores in each `samples/00000/result.json`. Reports include
accuracy, change in percentage points and current policy TPS. Historical baseline
TPS is marked historical; same-run TPS change/decode speedup are N/A. To generate
a fresh baseline explicitly later, pass `--baseline-mode fresh` to the Python CLI.
TPS sums visible answer tokens excluding EOS / synchronized decode seconds;
different answer lengths and forward counts are retained, not treated as equal
work. Reports update after each completed example. This remains development data.

Read `audio_cache/cache.py` for caching, `decode.py` for commitment/forward calls,
`model.py` for audio preparation, and `run.py` for the experiment. Other modules
handle the exact cohort, fixed answer extractor, reports and immutable checkout.

The runner can also compare the three cache schedules later:

```bash
source .venv/bin/activate
python -u -m audio_cache.run --policies 888 16816 161616 --out runs/cache_compare50
```

That is optional; the default launcher runs only16/16/16. Keep source,
inputs, sample count and settings unchanged when resuming. Use a fresh run name
after any change.

To stop the original v28 run and reuse its fully completed cache results:

```bash
pkill -TERM -f '[p]ython.*-m audio_cache[.]run' || true
git pull --ff-only origin audio-cache-clean
nohup env SKIP_INSTALL=1 REUSE_RUN=runs/cache161616_50_v28 \
  bash scripts/run.sh > cache161616_only50_v29.log 2>&1 &
tail -f cache161616_only50_v29.log
```

Wait for the old process to exit before pulling/restarting. `REUSE_RUN` validates
the input/model/decoder/seed/Torch configuration and copies completed cache
measurements into the new run; it does not overwrite old results. If the previous
run had another name, use that directory. Keep the same `REUSE_RUN` setting on
subsequent resumes. Incomplete examples without a completed policy are rerun.
