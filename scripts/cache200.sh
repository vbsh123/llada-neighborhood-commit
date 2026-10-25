#!/usr/bin/env bash
# Eight preselected cache schedules; reuse saved baselines by default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cohort_type=reused
if [[ "${FRESH_BASELINE:-0}" == 1 ]]; then
  [[ -z "${BASELINE_RUN:-}" ]] || { echo 'Choose FRESH_BASELINE or BASELINE_RUN, not both'; exit 2; }
  cohort_type=fresh
fi
export COHORT_MODE=cache200
export MANIFEST="${MANIFEST:-$PWD/data/cache_${cohort_type}200_seed${DATA_SEED:-20261025}_v31.jsonl}"
source scripts/setup.sh
run="${RUN_NAME:-cache_${cohort_type}200_v31}"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
baseline_args=(--baseline-mode fresh)
if [[ "$cohort_type" == reused ]]; then
  baseline_args=(--baseline-mode saved --saved-baselines "${manifest%.*}.baselines.json")
fi
python -u -m audio_cache.run --suite cache --manifest "$manifest" \
  --cohort-reference "${manifest%.*}.cohort.json" --out "runs/$run" \
  --samples "${SAMPLES:-200}" --diffa-root "$diffa_checkout" --verify-samples 0 --progress-every 32 \
  --policies 323232 161616 888 16816 16168 323216 32328 321632 "${baseline_args[@]}"
echo "CACHE200 COMPLETE: runs/$run/analysis/report.md"
