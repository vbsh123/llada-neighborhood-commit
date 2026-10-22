# Progressive global pruning and phased prompt-confidence priority

The two new modes run separately, each on **20 examples by default**, using the
existing 30-question manifest without duration filtering. `SAMPLES` overrides
this count. These are development/tuning results, not held-out validation.

## `gsm-progressive-prune`: four policies plus baseline

The audio policy starts acoustic removal at zero-based forward16 and semantic
removal at44. It never removes prompt text or answer positions. Scores average
all heads, **all transformer layers**, and all currently eligible masked answer
queries (current/earlier decoder blocks, excluding future blocks).

For each surviving prefix position compute `r=current_key_count*mean_attention`.
Uniform attention has r=1; eligibility requires **r<0.5 at two consecutive
observations**. Scores update every8 forwards, with an extra observation for
the44 boundary. Observations before a group's activation accumulate history.
At each decision remove the lowest-attention **ceil(25% of currently eligible
positions)** from each activated audio group. There is no token-count cap.
Fractional ceil permits the last eligible position to be removed. No arbitrary
one-token-per-group retention is enforced. All answer rows always remain.

To avoid looking ahead within a forward, observe indices7,15,23,... plus43,
and apply the completed decision at8,16,24,... plus44. The first acoustic event
can therefore apply at16, semantic at44. Actual activation is data-dependent.
An EOS before a queued event applies leaves that event unapplied and separately
logged. It does not count as a removal or execution saving.

Rows/KV columns are physically removed **before layer0** on the next forward
and stay absent throughout every later transformer block/forward. Original
absolute RoPE positions are preserved. Surviving rows recompute every forward;
no temporal KV/hidden cache banks. Embedding and final full vocabulary head
remain unchanged; the final block restores the full output shape once.

Four policies:

1. **audio**: threshold-qualified acoustic first; semantic from44. Prompt stays.
2. **early_text**: match every applied audio event's time and total count, but
   substitute prompt text for half the cumulative acoustic removal budget.
   Semantic removals retain the source count. Select the lowest-attention
   positions within each designated pool on this control's own trajectory.
3. **late_text**: acoustic source counts; substitute prompt text for half the
   cumulative semantic budget from44.
4. **generic**: same event times/counts, choose lowest attention over the entire
   surviving prefix.

Controls are reference-guided ablations, not independent deployable trigger
algorithms. Forced text substitution deliberately waives threshold/patience for
the chosen text count: otherwise the audio-derived threshold would remove
almost no prompt text. Cumulative halving avoids odd-batch rounding drift.
Prompt exhaustion is logged; the unmet replacement remains in its audio group,
preserving the total count. Controls do not use source position identities or
reference answers. Copy times/counts from the corresponding completed source
run; replay resolves its own controls from the paired replay source. Source
must precede controls; remaining policy ordering is shuffled reproducibly.

Correctness, baseline-correct losses/recoveries, safe-EOS forward counts and
clean synchronized autonomous latency are measured for all20 questions. Default
two timing repeats after warmup. **Equal-input replay defaults to the first2
questions**, up to256 forwards, two repeats. Replay asserts equal row/QK budgets
between all controls and the source. Monitoring/scoring, compaction and logging
bookkeeping are inside timings; JSON disk writes occur outside. The extra QK
scoring overhead may offset savings, and is part of the method's real cost.

`cache_benchmark.json` / `fixed_input_replay.json` accounting stores every
observation's positions, all-layer mean score, score/uniform, streak, query IDs,
eligible counts, requested/actual text replacement, and proposed batches.
Selections record actual apply step, preceding score step, removal identities,
composition, mass and counts. Warmup logs show activation/removal summaries;
timed-repeat completions are logged separately.

Calibration limits: threshold0.5 is motivated by earlier retrospective-query
baseline recordings, not proven safe for the current online query population.
Renormalized attention and changing key count can reinforce later removals.
The retained per-check scores and histories allow this to be inspected.

Settings: `ACOUSTIC_START=16`, `SEMANTIC_START=44`, `PRUNE_INTERVAL=8`,
`PRUNE_PATIENCE=2`, `PRUNE_THRESHOLD=0.5`, `PRUNE_ELIGIBLE_FRACTION=0.25`,
`BENCHMARK_REPEATS=2`, `REPLAY_SAMPLES=2`, `REPLAY_REPEATS=2`.

## `gsm-prompt-boost`: seven quality policies plus baseline

**Fixed prompt text, not revealed answer text.** All-layer/head attention from
currently eligible masked answer queries, as above. No caching/pruning combined.
Mean baseline EOS-stopped count from v19 is88 (20 examples, including failures).
Freeze the midpoint44 and end88; no per-test-example eventual length is used.

The dynamic temporal weights are:

`semantic_weight=max(0,1-t/44)`

`prompt_weight=clip((t-44)/44,0,1)`

Semantic bonus starts at forward0; it vanishes at44. Prompt starts at44 and
rises to full weight at88, staying there for longer answers. Both are zero at
exactly44. No attention-crossing gate or semantic-dominance prerequisite.

Within each current eligible query set and source group separately, compute
Q75/Q95 attention quantiles. `h=clip((attention-Q75)/(Q95-Q75+epsilon),0,1)`.
Below Q75 gets no bonus, >=Q95 gets full source factor. Flat attention gives0.
Use actual mean attention values for interpolation, not solely percentile rank.

Let c be original top1 probability and
`d=lambda*(semantic_weight*h_semantic+prompt_weight*h_prompt)`.

Adjusted priority is `sigmoid(logit(c)+d)`, implemented stably as
`c/(c+(1-c)*exp(-d))`, preserving exact0/1 and exact zero-shift identity.
The decoder adds the difference from original confidence for commitment ranking.
Logits, argmax token identity, block eligibility and commitment quota are unchanged.
Low attention gets no penalty. Priority is a heuristic, not calibrated confidence.

Test lambdas **0.05 and0.1**. Each has dynamic temporal weights, fixed semantic
weight1 throughout, and fixed prompt weight1 throughout. Include lambda0
observed baseline; visible output IDs must match normal baseline. **7 policies**,
one deterministic generation each; final numerical accuracy is paired against
baseline on all20. No speed claim: all-forward attention scoring is instrumented.

Every policy stores per-query source masses, Q75/Q95 thresholds, source factors,
original probabilities/IDs, log-odds shift, adjusted priority, actual bonus,
commitments, layer means and phase weights. Reports show nonzero-bonus sample
counts and each policy completion prints active bonus forwards, catching
inactivity rather than mistaking it for preserved-accuracy success.

Settings: `PRIORITY_STRENGTHS="0.05 0.1"`, `PRIORITY_MIDPOINT=44`, `PRIORITY_END=88`.
Baseline input token states are retained as compressed NPZ for the priority run.

## Running

Use new run names. Run jobs sequentially on one GPU. Existing setup suffices;
pull branch `audio-drift-study` then launch either new mode. Reports live under
`analysis/pruning_report.md`, `analysis/reasoning_report.md` (pruning), or
`analysis/attention_study_report.md` (boost), with their CSVs and per-sample JSON.
GPU correctness/latency must still be measured on Vast; local tests use a tiny
model built from the actual pinned upstream implementation.
