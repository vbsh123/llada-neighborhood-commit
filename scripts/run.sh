#!/usr/bin/env bash
# Default: only16/16/16, saved baseline, exact original50. One job, one GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup.sh
run="${RUN_NAME:-cache161616_only50_v29}"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
extra=()
if [[ -n "${REUSE_RUN:-}" ]]; then extra+=(--reuse-run "$REUSE_RUN"); fi
python -u -m audio_cache.run --manifest "$manifest" --out "runs/$run" \
  --policies 161616 --samples 50 --diffa-root "$diffa_checkout" \
  --baseline-mode saved --verify-samples 0 --progress-every 32 "${extra[@]}"
