#!/usr/bin/env bash
# Simulation only: same GSM8K questions in baseline, delay-only and frozen-state modes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-pilot}"
case "$mode" in
  smoke) count=2 ;;
  pilot) count=10 ;;
  study) count=100 ;;
  *) echo 'Usage: bash scripts/vast_sleep.sh [smoke|pilot|study]'; exit 2 ;;
esac
experiment_name="${RUN_NAME:-sleep_${mode}_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$experiment_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
experiment_out="runs/$experiment_name"
mkdir -p "$experiment_out"
exec > >(tee -a "$experiment_out/driver.log") 2>&1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python3 - <<'PY'
import sys
assert (3,10)<=sys.version_info[:2]<(3,13), 'Use Python 3.10–3.12'
PY
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -e '.[gpu]'
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA GPU required'
assert torch.cuda.is_bf16_supported(), 'BF16 GPU support required'
print('GPU:',torch.cuda.get_device_name(0))
PY
python -m unittest discover -s tests -p 'test_token_sleep.py' -v
python -m pip freeze > "$experiment_out/packages.txt"
nvidia-smi > "$experiment_out/nvidia-smi.txt"
input_args=()
if [[ -n "${SAMPLES_FROM:-}" ]]; then input_args+=(--samples-from "$SAMPLES_FROM"); fi
echo "SIMULATED SLEEP: $count questions × 3 modes; policy=${POLICY:-top1}; no speed claim."
python -u -m experiments.token_sleep.run \
  --out "$experiment_out" --samples "$count" --length 256 \
  --policy "${POLICY:-top1}" --commit-threshold "${COMMIT_THRESHOLD:-0.9}" \
  --high-entropy "${HIGH_ENTROPY:-2}" --sleep-confidence "${SLEEP_CONFIDENCE:-0.5}" \
  --high-sleep "${HIGH_SLEEP:-2}" --low-sleep "${LOW_SLEEP:-0}" \
  --influence-every "${INFLUENCE_EVERY:-1}" "${input_args[@]}"
bash scripts/export.sh "$experiment_out"
echo "Complete: $experiment_out"
