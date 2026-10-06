#!/usr/bin/env bash
# Run on Vast (CUDA/BF16 GPU). Never launch this script on the local CPU machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-smoke}"
case "$mode" in
  smoke) count="${SAMPLES:-2}"; length="${LENGTH:-64}" ;;
  pilot) count="${SAMPLES:-100}"; length="${LENGTH:-256}" ;;
  full) count="${SAMPLES:-500}"; length="${LENGTH:-256}" ;;
  *) echo 'Usage: bash scripts/vast.sh [smoke|pilot|full]'; exit 2 ;;
esac
experiment_name="${RUN_NAME:-window_${mode}_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$experiment_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
experiment_out="runs/$experiment_name"
mkdir -p "$experiment_out" .cache/matplotlib
exec > >(tee -a "$experiment_out/driver.log") 2>&1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
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
assert torch.cuda.is_bf16_supported(), 'BF16 support required'
print('GPU:',torch.cuda.get_device_name(0))
print('VRAM GiB:',torch.cuda.get_device_properties(0).total_memory/2**30)
PY
python -m unittest discover -s tests -v
python -m pip freeze > "$experiment_out/packages.txt"
nvidia-smi > "$experiment_out/nvidia-smi.txt"
# Resolve a common dataset commit before either policy runs, and retain it for
# resume. This avoids main vs resolved-hash config mismatches in the comparison.
if [[ ! -f "$experiment_out/dataset_revision.txt" ]]; then
  python - "${DATASET_REVISION:-main}" "$experiment_out/dataset_revision.txt" <<'PY'
from huggingface_hub import HfApi
from pathlib import Path
import sys
Path(sys.argv[2]).write_text(HfApi().dataset_info('openai/gsm8k',revision=sys.argv[1]).sha+'\n')
PY
fi
dataset_revision="$(cat "$experiment_out/dataset_revision.txt")"
for policy in top1 top1_window; do
  python -m confidence_geography.run \
    --out "$experiment_out/$policy" --policy "$policy" \
    --samples "$count" --length "$length" --block-length "${BLOCK_LENGTH:-0}" \
    --seed "${SEED:-1729}" --offset "${OFFSET:-0}" \
    --dataset-revision "$dataset_revision"
done
python -m confidence_geography.compare_window \
  --baseline "$experiment_out/top1" --window "$experiment_out/top1_window" \
  --out "$experiment_out/window_comparison.json"
bash scripts/export.sh "$experiment_out"
echo "Complete: $experiment_out"
