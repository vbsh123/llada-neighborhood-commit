#!/usr/bin/env bash
# CUDA inference only on Vast. Default: 10 matched examples per policy, never GSM8K/window/SOAR.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-pilot}"
case "$mode" in
  pilot) count=10 ;;
  smoke) count=2 ;;
  *) echo 'Usage: bash scripts/vast_translation.sh [pilot|smoke]'; exit 2 ;;
esac
experiment_name="${RUN_NAME:-translation_${mode}_$(date -u +%Y%m%dT%H%M%SZ)}"
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
  python -m pip install -e '.[gpu,translation]'
fi
python - <<'PY'
import torch,sacrebleu
assert torch.cuda.is_available(), 'CUDA GPU required'
assert torch.cuda.is_bf16_supported(), 'BF16 GPU support required'
print('GPU:',torch.cuda.get_device_name(0))
print('SacreBLEU:',sacrebleu.__version__)
PY
python -m unittest discover -s tests -p 'test_translation_probe.py' -v
python -m pip freeze > "$experiment_out/packages.txt"
nvidia-smi > "$experiment_out/nvidia-smi.txt"
python -u -m experiments.translation_probe.run \
  --out "$experiment_out" --samples "$count" --policies top1 threshold \
  --length "${LENGTH:-256}" --commit-threshold "${COMMIT_THRESHOLD:-0.9}"
bash scripts/export.sh "$experiment_out"
echo "Complete: $experiment_out"
