#!/usr/bin/env bash
# Seven conditions, sequentially on one GPU; saved baseline only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup.sh
name="${RUN_NAME:-audio_night50_v30}"
[[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
root="runs/$name"
count="${SAMPLES:-50}"
common=(--manifest "$manifest" --samples "$count" --diffa-root "$diffa_checkout" \
        --baseline-mode saved --verify-samples 0 --progress-every 32)
echo 'Phase1/3: cache32/32/32,32/16/32,32/32/16'
python -u -m audio_cache.run --suite cache --policies 323232 321632 323216 \
  --out "$root/cache" "${common[@]}"
python -m audio_cache.night_report --root "$root"
echo 'Phase2/3: global-attention pruning, audio and matched generic, no temporal cache'
python -u -m audio_cache.run --suite prune --out "$root/prune" "${common[@]}"
python -m audio_cache.night_report --root "$root"
echo 'Phase3/3: sparse audio attention, alone and with cache32/32/32'
python -u -m audio_cache.run --suite sparse --out "$root/sparse" "${common[@]}"
python -m audio_cache.night_report --root "$root"
echo "OVERNIGHT COMPLETE: $root/analysis/report.md"
