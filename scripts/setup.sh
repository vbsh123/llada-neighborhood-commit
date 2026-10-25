#!/usr/bin/env bash
# Shared setup only; does not start a generation experiment.
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
case "${COHORT_MODE:-development50}" in
  development50) python -m audio_cache.data --out "$manifest" ;;
  cache200)
    reference_args=()
    if [[ -n "${BASELINE_RUN:-}" ]]; then reference_args+=(--baseline-run "$BASELINE_RUN"); fi
    if [[ "${FRESH_BASELINE:-0}" != 1 ]]; then reference_args+=(--reuse-only); fi
    python -u -m audio_cache.prepare200 --out "$manifest" --seed "${DATA_SEED:-20261025}" "${reference_args[@]}"
    ;;
  *) echo 'Unknown COHORT_MODE'; exit 2 ;;
esac
