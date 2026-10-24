#!/usr/bin/env bash
# Three independent experiments; no older benchmark/diagnostic orchestration.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
suite="${1:-cache}"
case "$suite" in cache|prune|sparse) ;; *) echo 'Usage: bash scripts/vast_audio_clean.sh [cache|prune|sparse]'; exit 2;; esac
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then bash scripts/vast_audio_drift.sh setup; fi
audio_venv="${AUDIO_VENV:-$PWD/.venv-audio}"
source "$audio_venv/bin/activate"
pool="${DEV_POOL_MANIFEST:-$PWD/data/audio_drift/gsm_validation100_seed20261023_v21.jsonl}"
env MANIFEST="$pool" SAMPLES=50 POOL_SAMPLES=100 DATA_SEED=20261023 \
  DATASET_REVISION=e94caa2154433a54cd430702666c454977df3083 EXCLUDE_MANIFESTS= \
  bash scripts/vast_audio_independent.sh prepare
count="${SAMPLES:-50}"
run="${RUN_NAME:-audio_clean_${suite}${count}_v27}"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run name'; exit 2; }
options=()
if [[ "$suite" == prune ]]; then
  options+=(--query-scope "${QUERY_SCOPE:-all}" --prune-fraction "${PRUNE_FRACTION:-0.125}" \
    --prune-threshold "${PRUNE_THRESHOLD:-0.5}" --prune-layer-start "${PRUNE_LAYER_START:-0}")
  if [[ "${COMPARE_CURRENT_BLOCK:-0}" == 1 ]]; then options+=(--compare-current-block); fi
fi
python -u -m experiments.audio_clean.run --suite "$suite" --manifest "$pool" \
  --out "runs/$run" --samples "$count" --verify-samples "${VERIFY_SAMPLES:-2}" \
  --progress-every "${PROGRESS_EVERY:-32}" --cpu-threads "${CPU_THREADS:-4}" "${options[@]}"
echo "Complete: runs/$run"
