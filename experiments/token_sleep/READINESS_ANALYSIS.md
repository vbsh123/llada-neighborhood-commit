# Confidence buckets and downloaded sleep results

`analyze_readiness.py` is an offline analysis. It reads ZIPs or directories through
the existing `Source` reader; it never loads a model or changes any run results.

```bash
python -m experiments.token_sleep.analyze_readiness \
  --top1 /path/to/top1.zip \
  --threshold /path/to/threshold.zip \
  --sleep '/path/to/runs (4).zip' \
  --out /path/to/readiness_report
```

Requires NumPy and Matplotlib. The sleep ZIP can contain both completed sleep-only
runs. Use the full baseline traces for the first two arguments, not the thin
runtime benchmark traces.

Open `index.html`. All images/data are embedded, so this one file can be shared
without the PNG folder. Choose policy, confidence bucket and population. The
default population excludes observations committed in the same forward: these
positions could not have been scheduled to sleep. Delay 1 is commitment on the
next forward, delay 2 the forward after that. The alternate population includes
the current commitment forward as delay 0.

## Read the code in this order

- `generate_readiness_report`: overall collection, statistics, plots and exports.
- `collect_baseline`: for each question, read masked-position observations,
  record each actual commitment step, then attach commitment-step minus
  observation-step. All fresh observations contribute, not just first entries.
- `summarize`: 10-point confidence buckets, four entropy bands plus the combined
  population, exact empirical duration distributions and question-cluster CDF
  bootstrap intervals (1,000 replicates, seed 1729).
- `plot_readiness` and `write_html`: actual plotting and the standalone viewer.
- `sleep_diagnostics`: analyze actual assignment/wake/commit histories and score
  saved answers with the pinned harness components, separately from readiness.

## Definitions and limits

The baseline scope is final nonspecial answer positions before the final first
EOS/EOT. This is a retrospective analysis filter, not a decoder rule. The current
predicted token may still be special; the filter concerns the final position.
Entropy is full-vocabulary entropy in nats; confidence is the highest non-MASK
probability without renormalization. The same position can contribute to several
buckets as its confidence changes; positions with long lifetimes contribute more
snapshots. Percentages are pooled observation percentages. Bootstrap resampling
keeps each question's entire set of correlated snapshots together.

Ten-point confidence bins are not exact probability matching. Each table shows
mean confidence within the entropy band. Decoding progress and response position
are also not controlled. An association inside one coarse bucket does not alone
establish entropy's incremental predictive value. Small-count median bars are
hatched; those observations are retained in the data and tables.

Commitment is the model's chosen reveal, not reference correctness. Readiness
statistics describe the original, unchanged baseline trajectory. They cannot
establish how context would change under sleep or prove a safe sleep duration.

The actual sleep traces save entropy/confidence only when assigning a sleep.
Those selected observations must not be mixed into the baseline confidence
buckets as if every position were measured. Wake-to-commit distributions are
per sleep episode; repeated episodes at the same position are separate entries.
They describe the treatment's actual trajectory, not a full-computation
counterfactual. A nearby reveal metric counts only reveals whose changed input
arrives during a sleeping forward, excluding the reveal immediately before a
fresh wake; proximity is absolute token-index distance <=4. It is a descriptive
proxy, not a measured influence or a proposed proven-safe wake trigger.

## Outputs

- `index.html`: embedded graphs, controls, counts and differing-answer examples.
- `readiness.csv` / `readiness.json`: medians, means, sample counts, CDFs and
  question-cluster bootstrap intervals.
- `*_question_histograms.npz`: sufficient per-question duration histograms and
  confidence/entropy sums; avoids reparsing large baseline ZIPs for new views.
- `sleep_diagnostics.json`: harness scores, actual work/timing, episode and wake
  counts, and paired correctness between the new variants.
- `sleep_questions.json`: per-question diagnostics and original generated text.
- `baseline_cohorts.json`: baseline questions, references, generated answers and
  model revisions used for matching. These historical full traces are separate
  generations from the later vanilla timing experiment.
- `provenance.json`: analysis hash, cohort sizes and baseline accuracy. Input
  paths and definitions are also recorded in `readiness.json`.

All timing numbers retain the original full-window/post-stop scope. Reusing an
old baseline's accuracy requires matched questions/settings; old runtime is not
a controlled timing baseline on another machine.
