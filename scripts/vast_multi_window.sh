#!/usr/bin/env bash
# Separate region-window experiment. Default: treatment only, no vanilla run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-pilot}" in
  smoke) count=2 ;;
  pilot) count=10 ;;
  study) count=100 ;;
  *) echo 'Usage: bash scripts/vast_multi_window.sh [smoke|pilot|study]'; exit 2 ;;
esac
experiment_name="${RUN_NAME:-multi_window_${1:-pilot}_$(date -u +%Y%m%dT%H%M%SZ)}"
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
python -m unittest discover -s tests -p 'test_multi_window.py' -v
python -m pip freeze > "$experiment_out/packages.txt"
nvidia-smi > "$experiment_out/nvidia-smi.txt"
input_args=()
if [[ -n "${SAMPLES_FROM:-}" ]]; then input_args+=(--samples-from "$SAMPLES_FROM"); fi
if [[ "${ALLOW_SCALED_FP32_DRIFT:-0}" == 1 ]]; then input_args+=(--allow-scaled-fp32-drift); fi
if [[ "${COMPARE_ACTIVE:-0}" == 1 ]]; then input_args+=(--compare-active); fi
if [[ "${INCLUDE_ORIGINAL:-0}" == 1 ]]; then input_args+=(--include-original); fi
echo "MULTI WINDOW: $count questions; MASK gap=${REGION_GAP:-4}; padding=${RADIUS:-4}; refresh every ${REFRESH_EVERY:-8} forwards."
echo "Extra modes: active baseline=${COMPARE_ACTIVE:-0}; vanilla=${INCLUDE_ORIGINAL:-0}."
python -u -m experiments.multi_window.run \
  --out "$experiment_out" --samples "$count" --length "${LENGTH:-256}" \
  --seed "${SEED:-1729}" --policy "${POLICY:-top1}" \
  --commit-threshold "${COMMIT_THRESHOLD:-0.9}" \
  --radius "${RADIUS:-4}" --region-gap "${REGION_GAP:-4}" \
  --refresh-every "${REFRESH_EVERY:-8}" "${input_args[@]}"
# Keep this driver independent of older export scripts' local line endings.
mkdir -p exports
archive_path="exports/${experiment_name}_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar --exclude='*.partial' --exclude='*.tmp' --exclude='.lock' \
  -czf "$archive_path" -C runs "$experiment_name"
sha256sum "$archive_path" > "$archive_path.sha256"
echo "Export: $archive_path"
echo "Checksum: $archive_path.sha256"
echo "Complete: $experiment_out"
