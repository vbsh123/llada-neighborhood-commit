# DIFFA audio inference experiments

Standalone **orphan branch `audio-cache-clean`**: no inherited experiment tree or
imports from the older repo. Cache, global pruning and sparse attention each
have a separate small mechanism module.
Prior experiments and results remain on `audio-drift-study`.

## One overnight command:50 questions, seven conditions

After any active job finishes, update the checkout and launch:

```bash
cd /workspace/diffa-cache-clean
git pull --ff-only origin audio-cache-clean
nohup env SKIP_INSTALL=1 bash scripts/overnight.sh > audio_night50_v30.log 2>&1 &
tail -f audio_night50_v30.log
```

The launcher runs all phases **sequentially on one GPU**, with no interaction:

| Phase | Conditions | Questions each |
|---|---|---:|
| Cache |32/32/32,32/16/32,32/32/16|50|
| Global pruning |Audio only; matched generic fixed-prefix control|50|
| MARS-inspired sparse audio attention |Alone; combined with32/32/32 cache|50|

These are350 timed generations in total, plus one process warmup per condition.
**No uncached baseline is generated**, including warmup/verification. Saved
baseline outputs provide paired accuracy comparisons. No hidden representations,
full attention matrices, drift probes, replay, profiler passes or timing repeats.
The model loads once per phase. On a new machine omit `SKIP_INSTALL=1` to install.
The exact earlier50 questions/WAV hashes are verified before every phase.

The cache32 control from phase1 is used to assess sparse+cache32 in phase3 without
rerunning it. These are sequential same-GPU timings with independent generated
answers, not equal-input replay. Historical uncached TPS is explicitly marked
historical; same-run uncached speedups remain N/A.

Results are separate under `runs/audio_night50_v30/{cache,prune,sparse}`. The
combined `runs/audio_night50_v30/analysis/report.md` updates after each phase;
each phase's own report updates after each completed question. The final log
line is `OVERNIGHT COMPLETE: runs/audio_night50_v30/analysis/report.md`.
Use `RUN_NAME=...` for another output root. For a small unattended smoke run set
`SAMPLES=2`; default50. An unchanged rerun skips fully completed phases/questions.
Source/config changes require a new run name; earlier results are preserved.

### Global pruning details

`audio_cache/pruning.py` scores **every surviving query**, including audio,
fixed prompt, committed answer and all future/current MASK answer positions.
At each layer it averages actual softmax attention over all heads and queries;
then averages all32 layers. Self-attention is included. Per-key importance is
compared with uniform attention (`score * current_key_count`). Query chunks of32
bound memory without subsampling. The observation computation is timed.

Keep the previous gentle setting to isolate the query-scope change: observe
every8 forwards; score below0.5×uniform for two observations; acoustic removal
allowed from zero-based forward16 and semantic from44. Remove the bottom12.5%
of qualified surviving tokens per group/check, rounded up, with **no token cap**.
Apply decisions next forward and physically remove selected rows and KV columns
in **all32 layers**. Retained tokens keep original RoPE positions. No temporal
KV/hidden caching is mixed into pruning.

The generic control copies audio event times/removal counts but ranks its own
entire surviving prefix, including prompt text. Its threshold is waived to match
the source budget; it is reference-guided, not an independently tuned method.
Different EOS stopping can truncate applied events. Only compact group masses,
qualification counts, removal positions/events and work counts are saved.

### Sparse attention details

`audio_cache/sparse.py` implements a paper-inspired adaptation of
[MARS-Cache §4.2](https://arxiv.org/html/2601.17868v2#S4.SS2), not a reproduction
of an official released MARS kernel:

* First forward is dense. Anchor selection runs once at the first layer of each
  sparse group, using128 evenly spaced sequence queries and semantic-key softmax
  with diagonal entries excluded. Cache the indices for the remaining decode.
* Semantic audio is divided into chronological32-token chunks. Ordinary semantic
  queries see their own and adjacent chunks, global semantic anchors, and all
  non-semantic keys.
* Layers0–7 stay dense. Groups8–15,16–23,24–31 retain respectively50%,25%,12.5%
  semantic anchors per chunk, rounded up with minimumone.
* Acoustic summary queries and semantic-anchor queries keep full attention.
  Prompt/answer queries see all non-semantic keys plus semantic anchors. Acoustic
  keys remain globally visible; their indices are not assumed chronological.
* Separate gathered Q/K/V subsets feed SDPA. No full dense NxN mask is used to
  simulate savings. No tokens are deleted. Original RoPE coordinates remain.

Chunk/anchor budgets are initial engineering settings, **not validated audio
locality findings or parameters derived from our drift numbers**. Sparse attention
can reduce attention pairs yet lose quality or speed due to small kernel calls
and gathers. Actual A40 performance still needs measurement. Timing includes
anchor search and selection; accounting explicitly excludes search from pair
counts. Sparse and pruning are separate experiments and are not combined here.

## Existing cache-only launcher

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
