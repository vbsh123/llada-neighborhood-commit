# Stability and reuse in diffusion language models

Primary-source review, 2026-10-13. This is a targeted search, not an exhaustive
novelty assessment. No new inference was performed.

| Paper | Relevant mechanism | Relationship to our proposal |
|---|---|---|
| [dLLM-Cache](https://arxiv.org/html/2506.06295v1#S3.SS2.SSS2) | Compares current and cached response Value vectors; selectively refreshes changed tokens and reuses other KV/attention/FFN features, with periodic full refresh. | Temporal representation stability already guides token-level reuse. |
| [SPA-Cache](https://arxiv.org/html/2602.02544v1#S3.SS2) | Uses a low-dimensional Value-derived proxy for temporal drift, refreshing selected positions; adapts update budgets by layer. | Closest overlap with cheaper stability-based selection and expanding reuse. |
| [Elastic-Cache](https://arxiv.org/html/2510.14973v1) | Uses attention drift to choose refresh timing and depth; caches distant MASKs outside a sliding window. | Already decides when and where stale context should be refreshed. |
| [Window-Diffusion](https://arxiv.org/html/2601.20332v1) | Distinguishes active, buffer and far-field tokens. Buffer MASK KV is reused across a phase and refreshed periodically. | Sleeping masked-context tokens across several forwards is already present. |
| [EntropyCache](https://arxiv.org/html/2603.18489v1#S4) | Newly decoded token entropy triggers full refresh; partial forwards update current masks plus recent decoded positions. | Its entropy concerns revealed tokens and resulting cache drift, rather than using a masked position's entropy history to sleep it. It also uses decode recency. |
| [DynaTE](https://arxiv.org/html/2610.11284v1#S3.SS1) | High top-32 entropy, low confidence, distance and attention mass guide intra-forward query/FFN pruning; KV remains recomputed. | Direct overlap with uncertainty-based pruning, but different from multi-forward stale-KV reuse. |

## What the current data establish

Our observed duration-of-uncertainty and entropy-variation features offer modest
extra predictive value for future commitment timing overall. We have not isolated
the moderate-entropy band just below a pruning cutoff, and we have not measured
how entropy stability predicts KV stability or contextual error in that band.
Commitment timing is not a direct measure of safe representation reuse.

The broad ideas of temporal stability, stale state reuse, periodic refresh and
adaptive selection are therefore insufficient novelty claims by themselves.
An entropy-history rule specifically admitting additional moderate-entropy masked
positions remains a hypothesis to test, not an established literature gap.

## Next comparison aligned with the proposed speed improvement

1. Define a fixed baseline entropy metric and cutoff. DynaTE's top-32 truncated
   entropy cannot be directly compared with our full-vocabulary two-nat cutoff.
2. Isolate a band below that cutoff, rather than averaging high-entropy positions.
3. Compare candidates with similar current entropy/confidence/distance but
   different past histories, on held-out questions.
4. Measure near-future confidence rises and commitments as observational outcomes,
   plus KV drift or active-position prediction changes for reuse safety.
5. Actually reuse cached states and measure wall-clock savings and final accuracy
   against current-state-only and representation-drift caching baselines.

This review did not establish that existing papers test our exact band-specific
entropy-history rule. It also did not establish that no paper does.
