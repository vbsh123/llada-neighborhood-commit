# Removal sensitivity and attention-based commitment priority

Two separate modes, using the existing 30-question manifest without the old
two-clip duration filter. **Default 10 questions each**, override with `SAMPLES`.
No caching or persistent pruning is combined with priority. Runs used for tuning
are development data; retain baseline failures and use new held-out data later.

## Removal sensitivity: `gsm-sensitivity`

1. Run and verify baseline; retain cheap input-token states, not hidden vectors.
2. Locate first pre-forward states with >=25/50/75% final baseline lexical content
   revealed. Final content excludes EOS and special tokens. This checkpoint
   definition is retrospective analysis only. Short/empty answers and missing
   checkpoints are logged; duplicate checkpoints remain identifiable by step.
3. Observe online-eligible masked answer queries (current/earlier blocks, no
   future blocks). Average all heads and layers 25–31 by default. Rank prefix
   key positions by mean query attention.
4. Independently remove 8/16/32 acoustic, semantic, prompt-text or full-prefix
   positions. Evaluate both current ranking and the positions selected at the
   earliest available checkpoint. Budgets are equal counts, not fractions.
   Unsupported budgets are explicitly skipped. Every intervention starts from
   the untouched baseline state. Identical removal sets at the same checkpoint
   reuse measurements. The dense diagnostic removes KV columns for all queries
   in the selected layers; it is not an optimized timing path.
5. Primary metrics use the baseline lexical-content subset of eligible masked
   queries, excluding EOS/padding; all-eligible metrics are saved separately.
   No checkpoints after content completion. Empty content-query subsets have
   null metrics and are excluded from plot averages, with counts retained.
   Save per-query KL(baseline||perturbed), top1 flips, baseline-top1 probability
   loss, baseline entropy, identities, query counts, removed mass/composition.
   Query populations change across checkpoints; fixed token selection controls
   removed positions, not query difficulty. Full continuation accuracy is a later
   follow-up after inspecting these forward-only measurements.

Settings: `SENSITIVITY_FRACTIONS="0.25 0.5 0.75"`,
`SENSITIVITY_BUDGETS="8 16 32"`, `SENSITIVITY_LAYER=25` (0 for all layers).
Default 72 measurement records per supported example, plus 3 reference forwards;
duplicate intervention reuse reduces actual model calls. Each pool completion
is logged so the diagnostic does not remain silent for an entire example.

## Commitment priority: `gsm-attention-priority`

This is a generation-quality experiment, not calibrated confidence estimation.
At each forward collect attention **from all currently eligible masked answer
positions**, averaged across **all heads and all 32 layers**. Source attention is
summed over semantic audio, fixed prompt text, acoustic audio, and already
revealed answer text. Revealed text excludes special tokens and positions at/
after the first revealed EOS. No future answer or reference information is used.

Main text source: `PRIORITY_TEXT_SOURCES=revealed`. Use `prompt` for fixed prompt
text or `"revealed prompt"` to run both separately (13 policies rather than 7).
This differs from the earlier graph's fixed prompt and retrospective query set.

Starting at **zero-based forward 40**, smooth eligible-query group means with
EMA(new weight .2). Earlier observations do not seed the EMA. Define
`margin=(semantic-text)/(semantic+text)`. Arm the transition only after observing
positive margin on/after40. If text already dominates at40, wait; this is not an
immediate cutoff. A later nonpositive margin is the cutoff. It need not occur.

Before cutoff the semantic coefficient follows the running minimum of
`clip(margin/initial_positive_margin,0,1)`. At cutoff it becomes zero permanently.
After cutoff the text coefficient follows the running maximum of
`clip(-margin,0,1)`. These monotone coefficients follow the smoothed attention
balance, not a predetermined forward ramp or a numerical time derivative.
An EMA at sampled forward0 does not imply global causality; eligible queries and
revealed key count change over time. Revealed-key count is logged to expose this.

For each eligible query, normalize its semantic and text group masses by the
maximum source mass among current eligible queries, yielding saliencies [0,1].
`priority = original_top1_probability + alpha*(semantic_weight*semantic_saliency
                                           + text_weight*text_saliency)`.
Logits, top1 predicted IDs, masks/block eligibility, and commits per forward are
unchanged. Only the ranking used to select committed positions changes. This
priority may exceed1; it is not a probability and is never reported as confidence.

Default strengths `PRIORITY_STRENGTHS="0.02 0.05"` are exploratory: up to .02 or
.05 additive priority, not a percent relative probability increase. For each
strength run dynamic transition, fixed semantic-only bonus from40, and fixed
text-only bonus from40. Include an attention-observed alpha0 identity control,
which must reproduce baseline visible IDs exactly. **7 policies plus the normal
baseline**, one deterministic decode each; no timing repeats or replay. No speed
claim: observed times include full attention-scoring/logging overhead.

Logs retain every forward's all-layer per-query group masses, layer means,
original top1 probabilities and IDs, bonuses, actual commitments, EMA/coefficient
state, and cutoff. The global cutoff uses an equally weighted layer/head/query
mean, not individual-token crossings. Dynamic and fixed policies compute their
own online attention on their own generation trajectories.
Per-forward EMA semantic/text change rates and balance change rates are also
logged, so the relation between coefficient evolution and attention changes can
be inspected directly. Strength follows the normalized balance, not a raw
derivative, whose scale would depend on sampling frequency.

## Outputs

`samples/NNNNN/baseline_token_states.npz`: full baseline input token states;
`baseline.json`, `steps.json` preserve output and commit history. Sensitivity:
`removal_sensitivity_config.json`, flushed `removal_sensitivity.jsonl`. Priority:
`attention_priority.json` and one detailed JSON per policy, written after each
completed policy. Standard incomplete-folder persistence rules apply.

`analysis/removal_sensitivity.csv` and `.png`: equal-example mean KL by requested
checkpoint, pool and budget; consult CSV for missing checkpoints and query counts.
`analysis/attention_priority_pairs.csv`, `attention_priority_trajectories.csv`,
`attention_study_report.md`: quality, cutoff coverage, masses and weights.
Unmasking remains fixed-quota, not threshold-variable. Prioritizing highly
attending tokens is a hypothesis about useful commit order, not proof of causal
dependence. A successful quality heuristic still needs broader held-out testing.
