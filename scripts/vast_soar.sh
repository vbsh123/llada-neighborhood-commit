#!/usr/bin/env bash
# SOAR measurements only. Run inference on Vast, not the local CPU machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-smoke}"
case "$mode" in
  setup) ;;
  smoke) count="${SAMPLES:-2}" ;;
  pilot) count="${SAMPLES:-100}" ;;
  *) echo 'Usage: bash scripts/vast_soar.sh [setup|smoke|pilot]'; exit 2 ;;
esac
soar_checkout="${SOAR_ROOT:-$PWD/external/SOAR}"
python3 experiments/soar_reuse/checkout.py --soar-root "$soar_checkout"
if [[ "$mode" == setup ]]; then exit 0; fi
experiment_name="${RUN_NAME:-soar_support_${mode}_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$experiment_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -e '.[gpu]'
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA GPU required'
assert torch.cuda.is_bf16_supported(), 'BF16 support required'
print('GPU:', torch.cuda.get_device_name(0))
PY
python -m unittest discover -s experiments/soar_reuse -p 'test_*.py' -q
echo "Standalone SOAR: $count GSM8K questions, beam=${BEAM_SIZE:-2}, length=${LENGTH:-256}"
python -u experiments/soar_reuse/run_probe.py \
  --soar-root "$soar_checkout" \
  --out "runs/$experiment_name" --samples "$count" --beam-size "${BEAM_SIZE:-2}" \
  --length "${LENGTH:-256}" --block-length "${BLOCK_LENGTH:-0}" \
  --steps "${STEPS:-0}" --seed "${SEED:-1729}" --offset "${OFFSET:-0}"
python -m pip freeze > "runs/$experiment_name/packages.txt"
nvidia-smi > "runs/$experiment_name/nvidia-smi.txt"
echo "Complete: runs/$experiment_name/summary.json"
