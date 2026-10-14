#!/usr/bin/env bash
# Real sparse execution: identical backend without sleep versus real sleep.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-pilot}" in
  smoke) count=2 ;;
  pilot) count=10 ;;
  study) count=100 ;;
  *) echo 'Usage: bash scripts/vast_sleep_benchmark.sh [smoke|pilot|study]'; exit 2 ;;
esac
experiment_name="${RUN_NAME:-real_sleep_${1:-pilot}_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$experiment_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
experiment_out="runs/$experiment_name"
mkdir -p "$experiment_out"
exec > >(tee -a "$experiment_out/driver.log") 2>&1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -e '.[gpu]'
fi
python - <<'PY'
import sys,torch,numpy,pyarrow,datasets
assert (3,10)<=sys.version_info[:2]<(3,13), 'Use Python 3.10–3.12'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 CUDA GPU required'
print('GPU:',torch.cuda.get_device_name(0))
PY
python -m unittest discover -s tests -p 'test*token_sleep.py' -v
python -m pip freeze > "$experiment_out/packages.txt"
nvidia-smi > "$experiment_out/nvidia-smi.txt"
input_args=()
if [[ -n "${SAMPLES_FROM:-}" ]]; then input_args+=(--samples-from "$SAMPLES_FROM"); fi
conditions=2
if [[ "${INCLUDE_ORIGINAL:-0}" == 1 ]]; then input_args+=(--include-original); conditions=3; fi
echo "REAL SLEEP: $count questions × $conditions modes; GPU parity checks required before timing."
python -u -m experiments.token_sleep.benchmark \
  --out "$experiment_out" --samples "$count" --length 256 \
  --policy "${POLICY:-top1}" --commit-threshold "${COMMIT_THRESHOLD:-0.9}" \
  --high-entropy "${HIGH_ENTROPY:-2}" --sleep-confidence "${SLEEP_CONFIDENCE:-0.5}" \
  --high-sleep "${HIGH_SLEEP:-2}" "${input_args[@]}"
bash scripts/export.sh "$experiment_out"
echo "Complete: $experiment_out"
