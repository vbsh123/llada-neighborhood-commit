#!/usr/bin/env bash
# Run from the repo checkout on Vast. Uses its own environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-smoke}"
case "$mode" in
  setup|smoke|pilot) ;;
  *) echo 'Usage: bash scripts/vast_audio_drift.sh [setup|smoke|pilot]'; exit 2 ;;
esac
audio_venv="${AUDIO_VENV:-$PWD/.venv-audio}"
if [[ ! -x "$audio_venv/bin/python" ]]; then
  "${PYTHON:-python3}" -m venv "$audio_venv"
fi
source "$audio_venv/bin/activate"
python - <<'PY'
import sys
assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Use Python 3.10–3.12'
PY
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install "torch==${TORCH_VERSION:-2.6.0}" "torchaudio==${TORCHAUDIO_VERSION:-2.6.0}" \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
  python -m pip install -r experiments/audio_drift/requirements.txt
  python -m pip check
fi
diffa_checkout="${DIFFA_ROOT:-$PWD/external/DIFFA}"
python -m experiments.audio_drift.checkout --diffa-root "$diffa_checkout"
if [[ "$mode" == setup ]]; then exit 0; fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA GPU required'
assert torch.cuda.is_bf16_supported(), 'BF16-capable GPU required'
print('GPU:', torch.cuda.get_device_name(), 'VRAM GB:', torch.cuda.get_device_properties(0).total_memory / 1e9)
PY
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
experiment_name="${RUN_NAME:-audio_drift_${mode}_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$experiment_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
if [[ "$mode" == smoke ]]; then
  manifest="${MANIFEST:-$PWD/data/audio_drift/smoke.jsonl}"
  if [[ ! -f "$manifest" && -z "${MANIFEST:-}" ]]; then
    python -m experiments.audio_drift.prepare --out "$manifest" --samples 2
  fi
  count="${SAMPLES:-2}"
  length="${LENGTH:-32}"
  block="${BLOCK_LENGTH:-16}"
else
  : "${MANIFEST:?Set MANIFEST to your ASR/QA/prosody JSONL manifest; see experiments/audio_drift/README.md}"
  manifest="$MANIFEST"
  count="${SAMPLES:-100}"
  length="${LENGTH:-128}"
  block="${BLOCK_LENGTH:-32}"
fi
extra=()
if [[ "${INTERVENTIONS:-0}" == 1 ]]; then extra+=(--interventions freeze remove text_only); fi
if [[ "${TOKEN_METRICS:-0}" == 1 ]]; then extra+=(--token-metrics); fi
python -u -m experiments.audio_drift.run \
  --diffa-root "$diffa_checkout" --manifest "$manifest" --out "runs/$experiment_name" \
  --samples "$count" --length "$length" --block-length "$block" --steps "${STEPS:-$length}" \
  --layers "${LAYERS:-all}" --reference-device "${REFERENCE_DEVICE:-cpu}" \
  --verify-samples "${VERIFY_SAMPLES:-1}" --seed "${SEED:-1729}" "${extra[@]}"
python -m experiments.audio_drift.analyze --run "runs/$experiment_name"
python -m pip freeze > "runs/$experiment_name/packages.txt"
nvidia-smi > "runs/$experiment_name/nvidia-smi.txt"
echo "Complete: runs/$experiment_name/summary.json"
