# Cache plus progressive audio pruning

Run both schedules on the same questions, sequentially on one GPU. Each has a
cache-only control, giving four policies plus the paired clean baseline:

* Acoustic/prompt/semantic refresh 16/8/16, with and without audio pruning.
* Acoustic/prompt/semantic refresh 8/8/8, with and without audio pruning.

No confidence boost or text pruning. Pruning retains the v21 settings: check
every 8 forwards, require two low-attention observations, threshold 0.5 times
uniform attention, remove the lowest-attention quarter of qualifying positions
per group/check. Acoustic starts at forward index16, semantic at44. All layers
participate. Scores use each combined policy's own cached trajectory; decisions
take effect on the next forward. Removed rows never revive during a refresh.

Cache skips QKV/query/MLP updates for surviving prefix rows on nonrefresh
forwards. Pruning additionally excludes removed KV columns at every layer.
The combination uses PrefixCache's full absolute-coordinate backing banks and
hidden canvas, rather than the standalone pruning engine's compact interlayer
canvas. Removed bank entries remain allocated but are not read by attention.
Monitoring, selection, gathers/scatters and cache initialization are all timed.
This implementation detail means gains must be measured, not multiplied from
the previous standalone speedups.

On an existing configured Vast checkout:

```bash
git pull --ff-only origin audio-drift-study
nohup env SAMPLES=50 POOL_SAMPLES=100 REFERENCE_DEVICE=cuda VERIFY_SAMPLES=2 \
  bash scripts/vast_audio_independent.sh combined > combined50_v22.log 2>&1 &
tail -f combined50_v22.log
```

For a fresh environment, replace `vast_audio_independent.sh` with
`vast_audio_unattended.sh`; setup and model/data downloads then run unattended.
The repository is private, so cloning/pulling requires your GitHub credentials.

The default seed and100-question pool reproduce v21. The first50 questions have
already been inspected: this is a paired development comparison, not fresh
held-out validation. Preparation checks exact inputs and excludes the original
30 development IDs; it does not claim the v21 pool is still unseen. Use a new
RUN_PREFIX rather than modifying an existing run's settings to resume.

Output: `runs/audio_independent_combined50_v22`. Accuracy, synchronized warmed
decode timings (two repeats), forward counts, visible token counts, pruning
decisions and per-layer work counts are saved. Equal-input replay uses the first
two questions by default, with up to256 identical baseline states. Its outputs
are not graded. Frozen secondary final-answer grading runs automatically.
Compare combined policies with their cache-only partners and clean baseline;
retain baseline failures and all examples in primary results.
