# SOAR discarded-branch reuse probe

Observation only. No merges, confidence boosting, changed branch scores, or
extra validation/model forwards. Actual accuracy/speed gains from merging
cannot be measured until a merge policy is implemented and compared.

Official source: https://github.com/duterscmy/SOAR . Commit
`ec3eb400e41a43dc05db20a49c0219b9a968d28e`, LLaDA `generate_soar()`.
The full official repository is cloned into `external/SOAR` and checked out
at that commit (detached HEAD). The checkout is ignored by our Git repo; it is
not vendored or modified. The probe verifies HEAD, a clean working tree, and
the decoder SHA256 before loading model weights. It then instruments the
source in memory using AST hooks. `hooks.py:instrument()` inserts observers before
expansion, after each candidate append, and after pruning. The upstream
selection, scoring, deduplication and collapse rules remain unchanged.

Run on Vast with the existing environment/checkpoint cache and saved run:

```bash
git pull
bash scripts/vast_soar.sh setup
source .venv/bin/activate
python -u experiments/soar_reuse/run_probe.py \
  --window runs/window4_repair75_500_v1/top1_window \
  --out runs/soar_reuse_smoke_v1 --samples 2 --beam-size 2
```

After the smoke finishes, a 100-question measurement pilot:

```bash
python -u experiments/soar_reuse/run_probe.py \
  --window runs/window4_repair75_500_v1/top1_window \
  --out runs/soar_reuse_pilot_v1 --samples 100 --beam-size 2
bash scripts/export.sh runs/soar_reuse_pilot_v1
```

This runs SOAR only, not top1/window/repair again. It reuses the source sample
order, exact prompt IDs, model revision, response length and block length. It
does not need another dataset download. Model inference has not been run locally.
The source run is never written. Resume unchanged commands to skip completed
questions. Progress prints every 32 decoding steps and after each question.

## Measurements

Each `samples/ID/branches.jsonl.gz` includes all evaluated parent states, top1
predictions/confidence/full-vocabulary entropy at active-block masked positions,
candidate ancestry and new commits, scores, exact-state duplicate/pruned/retained
statuses, retained IDs, and potential reuse events. Indices are response-relative.

A reuse event is a position masked in the best survivor but filled by a unique
pruned candidate. Ignore special donor tokens, exact-state duplicates and tokens
already held by another surviving branch. Count each position once per step,
regardless of donor count. Historical commitments in pruned states are included,
not just commits made at that step. Repeated proposals on later steps count as
separate events; each question also reports unique position counts.

Report progressively stricter subsets:

1. All proposing donors agree on the token (one donor is vacuously unanimous;
   this is NOT evidence of agreement between multiple independent branches).
2. The best survivor's evaluated parent also predicts that token while masked.
   Its confidence is logged, without a hidden confidence gate.
3. At least one proposing donor state has no conflicting revealed token with
   the best survivor. This is syntactic compatibility, not semantic validity.

Final-match rates use only positions before the chosen final output's first
stop. This is a retrospective filter, not an online selection rule. Matching
the final output is consistency, not a reference correctness label. Numeric
answer accuracy is scored separately. No speedup is claimed from counting
tokens or events; validation can cost more than merging saves.

At positions still masked AND eligible in every evaluated parent (at least two),
compute uniform-mixture entropy, mean branch entropy and their difference
(mutual information between constructed branch ID and token). Full-vocabulary
probabilities are computed transiently in FP32 chunks; only scalar metrics are
saved. This is not entropy over model weights or independent posterior samples.
Beam scores are not used as mixture probabilities. Top1 agreement is logged
separately from mixture metrics. Single-parent steps have no mixture estimates.

## Explicit protocol choices

- Upstream threshold stays **strictly >0.95**, max parallel tokens **5**,
  min parallel tokens **1**, cumulative score is **sum of probabilities**, and
  beam collapse/deduplication rules remain upstream. This differs from our
  previous threshold-0.9 collector.
- Temperature 0, CFG 0, no cache; prompt/model come from the source run.
- Maximum iterations set to response length (256 for this experiment), rather
  than upstream default 128. No early stop on EOS. Final scoring trims first
  EOS/eot using the saved stop IDs and the existing numeric-answer extractor.
- Upstream MASK and special-token selection behavior is retained. Unlike our
  window collector, MASK is not explicitly excluded from upstream predictions.
  Report unresolved MASK counts, rather than silently repairing incomplete runs.
- Branches are batched. Report BOTH model calls and evaluated branch-sequence
  counts. A two-branch call is not equivalent in cost to one single-branch call.
- All logging/statistics add overhead. GPU forward time is separately measured;
  elapsed time includes probe overhead and excludes model loading. No merging
  decoder, speed improvement or paper reproduction claim is made.
- More VRAM may be required than window generation due to beam batching. Start
  with the two-question smoke. No automatic fallback to another beam size.

## Read the code

`hooks.py:reuse_opportunities()` defines hypothetical reusable tokens.
`run_probe.py:Probe.before()` captures predictions and mixture statistics.
`Probe.candidate()` records ancestry. `Probe.after()` records pruning and donor
support. `Probe.summary()` computes counts and retrospective consistency.

`result.json` includes a token dictionary for reading examples. Opportunity
percentages use all SOAR steps (or all pruning steps) as their denominator;
proposal events themselves are restricted retrospectively to pre-final-stop
positions. State-conflict checks compare full response states, including stop
and post-stop slots; they are deliberately labeled syntactic compatibility.

CPU tests:

```bash
python -m unittest discover -s experiments/soar_reuse -p 'test_*.py' -v
```

## Cross-branch support and future convergence (new)

There are two separate measurements. Neither changes SOAR's decisions.

**Discarded revealed-token support:** for every eligible donor token in a reuse
opportunity, record the best retained candidate's evaluated parent's probability
for that exact token at that exact position. This does not require it to be that
parent's top1. Donor confidence is historical confidence at commitment; recipient
support is from this forward. The output `discarded_donor_support_sweep` pools
ordered donor observations (multiple donors at one position count separately).
For each donor-confidence cutoff, the denominator is all observations with
known donor confidence above that cutoff and a recipient probability; the
numerator additionally passes the recipient-support cutoff. No reference answer
is used to decide which observations qualify.

**Different top1 predictions:** at every forward, compare ordered pairs A -> B
at positions that are still masked and eligible in both. If A's per-position
vocabulary top1 token differs from B's, record A's confidence and B's probability
for A's token. A need not select that position for commitment. Special donor
predictions are excluded. This is always the SAME response position, not matching
common words anywhere in the output.

Follow B's actual retained descendants through candidate ancestry, including
branch splitting and beam reordering. For each observation record:

- First later forward where any retained descendant predicts A's token as top1
  while that position is still masked and eligible.
- First forward whose retained descendant commits that token. Delay zero means
  commitment during the original comparison forward.
- Highest subsequent masked-position support among those descendants.
- Whether B's lineage disappears; an extinct lineage has no observable future.
- Whether B lies on the final winning lineage, and if so whether its final token
  at that position matches A's prediction. An unrelated survivor's matching
  token does NOT count as convergence of B.

`convergence.jsonl.gz` stores one event per ordered pair/position/forward;
repeated disagreements at the same position count again. `branches.jsonl.gz`
adds `cross_support` to masked-position rows: probabilities for tokens previously
predicted there, allowing probability trajectories to be reconstructed without
saving the full vocabulary. The scalar summaries do not duplicate every
trajectory point, to keep output size manageable.

`summary.json` contains `convergence_threshold_sweep`: donor confidence cutoffs
0.75, 0.85, 0.9, 0.95 crossed with recipient-support cutoffs 0.05, 0.1, 0.2, 0.3,
0.5, 0.75, 0.9. Both cutoffs are inclusive. Raw probabilities are saved so other
thresholds can be analyzed later. Top1/commit rates divide by all qualifying
observations; extinct recipients remain in that denominator and are separately
counted. Final-match rates divide ONLY by observations whose recipient is on the
actual final winning lineage. Statistics exclude positions beyond the actual
final first stop retrospectively; this does not alter decoding. Agreement and
convergence do not establish correctness or safe merging.

Read `convergence.py:measure_convergence()` for the ancestry/counting logic;
`run_probe.py:Probe.before()` collects probabilities and
`Probe.after()` collects exact-token donor support. The pooled counts are built
at the end of `run_probe.py:run()`.

Use a fresh output folder for this version, for example:

```bash
python -u experiments/soar_reuse/run_probe.py \
  --window runs/window4_repair75_500_v1/top1_window \
  --out runs/soar_support_convergence_smoke_v1 --samples 2 --beam-size 2
```

It adds probability-scoring/logging and CPU ancestry-analysis overhead, but no
extra model forwards. GPU execution still needs the Vast smoke test. The probe
code fingerprint includes `convergence.py` and `checkout.py`, preventing mixed-version resumes.


## Vast wrapper with the cloned upstream repo

On an existing configured machine, from this experiment repo:

```bash
git pull
bash scripts/vast_soar.sh setup
WINDOW_RUN=runs/window4_repair75_500_v1/top1_window \
RUN_NAME=soar_checkout_smoke_v1 SKIP_INSTALL=1 \
  bash scripts/vast_soar.sh smoke
```

Then, after inspecting the two-question smoke:

```bash
WINDOW_RUN=runs/window4_repair75_500_v1/top1_window \
RUN_NAME=soar_checkout_pilot_v1 SKIP_INSTALL=1 \
  bash scripts/vast_soar.sh pilot
```

On a fresh machine, clone our repo and restore the completed source window run
at the path passed to `WINDOW_RUN`. Omit `SKIP_INSTALL=1` to create the venv and
install our existing pinned GPU dependencies. It does not regenerate window or
top1 outputs and does not download GSM8K. Checkpoint weights may need downloading
if absent. The upstream repo's complete evaluation environment is not installed:
we call its pinned LLaDA decoder using our existing model environment, as before.

`setup` only clones/verifies SOAR: no packages, dataset, model, or inference.
`smoke` runs 2 questions; `pilot` runs 100. Both retain the source response length
(256 here), use beam size 2, and retain upstream strict >0.95 threshold and
maximum parallel commits 5. Set `SAMPLES`, `BEAM_SIZE`, `RUN_NAME`, or `SOAR_ROOT`
explicitly to change them. No Hugging Face cache variables are forced.

The same script can resume an unchanged `RUN_NAME`; completed questions are
skipped. Use a new output folder after changing probe code or source settings.
The earlier single-file-download manifests cannot be resumed by this version.
`summary.json`, per-question `result.json`, `branches.jsonl.gz`, and
`convergence.jsonl.gz` are under `runs/RUN_NAME`. Export manually if desired:

```bash
bash scripts/export.sh runs/soar_checkout_pilot_v1
```

An existing upstream checkout is never reset, pulled, or overwritten. A wrong
commit or dirty checkout produces a clear error. Set `SOAR_ROOT` to another
new directory if you want to keep a checkout containing your own edits. The
probe itself never writes inside the SOAR checkout. Read their code at
`external/SOAR/eval_llada8b/generate.py`, and our checkout checks in `checkout.py`.
