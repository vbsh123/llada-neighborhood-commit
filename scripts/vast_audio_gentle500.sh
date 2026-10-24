#!/usr/bin/env bash
# Same500 held-out questions and gentle pruning; choose full stack or deep suffix.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
variant="${1:-}"
case "$variant" in
  all) export PRUNE_LAYER_START=0;;
  deep) export PRUNE_LAYER_START=25;;
  *) echo 'Usage: bash scripts/vast_audio_gentle500.sh [all|deep]'; exit 2;;
esac
export SAMPLES="${SAMPLES:-500}"
export PRUNE_FRACTION=0.125
export MINIMAL_RECORDS=1
export RUN_NAME="${RUN_NAME:-audio_prune${SAMPLES}_gentle_${variant}_v25}"
exec bash scripts/vast_audio_prune500.sh
