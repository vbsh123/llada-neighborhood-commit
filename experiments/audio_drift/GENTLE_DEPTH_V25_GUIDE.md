# Gentle pruning on500 held-out questions: full stack versus deep suffix

Two independent single-GPU jobs, each with baseline, audio-only pruning and
reference-guided matched-budget generic pruning. The configurations use the
same500 heldout_v23 inputs, excluding all130 previously inspected questions.
No caching, confidence boost, sweep, repeated timing or replay.

Freeze the development setting that scored26/50 audio versus24/50 generic:
threshold0.5 times uniform attention, remove12.5% of currently qualifying tokens
per check, patience2, checks every8 forwards, acoustic start16 and semantic44.
There is no fixed token-count cap. All answer positions remain active.

* `all`: physical row/KV-column removal from layer0 through31.
* `deep`: layers0–24 keep every token and compute fresh states every forward.
  Selected rows/KV columns are physically removed before layer25 and stay absent
  through31 on that forward. Earlier layers receive them again on the next
  forward; the selected exclusion set remains in effect at the deep boundary.

The same observation rule uses all32 layers in both jobs. In deep mode key counts
differ across layers, so attention relative to uniform is normalized using each
layer's actual key count before averaging. Raw attention means rank candidates;
only positions still present in the deep suffix can qualify for additional
removal. Decisions finish before being applied on the next forward. Original
RoPE coordinates are preserved. The generic control copies its own run's audio
event times/counts but ranks its own prefix; it is not independently tuned.
Different depth trajectories can produce different removal amounts, so this is
a comparison of the deployed policies, not a matched-budget depth ablation.

Both jobs generate once per condition/question, with512 slots,32-token blocks,
256 maximum forwards and safe EOS stopping. Each process warms once. First two
baseline visible answers are checked against the official full decoder outside
decode timings. Total TPS uses visible answer tokens excluding EOS divided by
summed synchronized decode seconds. Attention observation/selection costs are
included; model loading, audio preparation, scoring and disk writes are excluded.

Minimal output includes texts/IDs, original/frozen secondary scores, times,
tokens, forwards, layer work counts, activation and actual removal decisions.
No per-position attention histories or per-forward commitment files. The internal
attention computations required for pruning still run across all layers.

On machine A (existing configured checkout):

```bash
cd /workspace/llada-audio-drift
git pull --ff-only origin audio-drift-study
nohup env SKIP_INSTALL=1 bash scripts/vast_audio_gentle500.sh all \
  > audio_gentle_all500_v25.log 2>&1 &
tail -f audio_gentle_all500_v25.log
```

On machine B, clone branch audio-drift-study and run:

```bash
nohup env SKIP_INSTALL=0 bash scripts/vast_audio_gentle500.sh deep \
  > audio_gentle_deep500_v25.log 2>&1 &
tail -f audio_gentle_deep500_v25.log
```

SKIP_INSTALL=0 includes setup unattended; use1 for an already configured machine.
Use matching GPU models/software if possible and compare speed against each
job's own baseline. Estimated A40 runtime is about8–12hours per job, depending on
new answer trajectories and audio lengths; downloads/setup add time.
Never pull changed source during an active job or run both jobs on one GPU.

Outputs: `runs/audio_prune500_gentle_all_v25` and
`runs/audio_prune500_gentle_deep_v25`. Compare printed INPUT_FINGERPRINT and
`input_fingerprint.json`: they must match exactly. Defaults reproduce the
heldout_v23 manifest automatically on a new machine. Baseline visible outputs
should also match across machines. Rerun an identical command to resume completed
examples; changed source/configuration requires a new output directory.

Read `analysis/lean_report.md`, `lean_pairs.csv`, original reasoning/pruning
reports, and frozen secondary extraction protocol/scores. Legacy median_seconds
holds one duration. All baseline failures and nonactivated cases remain included.
The deep policy is a new held-out candidate, not yet a validated improvement.
