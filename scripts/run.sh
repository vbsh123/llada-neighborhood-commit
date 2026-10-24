#!/usr/bin/env bash
# Default: baseline +16/16/16, exact original50. One job, one GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
audio_venv="${AUDIO_VENV:-$PWD/.venv}"
if [[ ! -x "$audio_venv/bin/python" ]]; then "${PYTHON:-python3}" -m venv "$audio_venv"; fi
source "$audio_venv/bin/activate"
python -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python3.10–3.12"'
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -r requirements.txt
  python -m pip check
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
diffa_checkout="${DIFFA_ROOT:-$PWD/external/DIFFA}"
python -m audio_cache.checkout --diffa-root "$diffa_checkout"
manifest="${MANIFEST:-$PWD/data/development50.jsonl}"
python -m audio_cache.data --out "$manifest"
run="${RUN_NAME:-cache161616_50_v28}"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
python -u -m audio_cache.run --manifest "$manifest" --out "runs/$run" \
  --policies 161616 --samples 50 --diffa-root "$diffa_checkout" \
  --verify-samples "${VERIFY_SAMPLES:-2}" --progress-every 32
