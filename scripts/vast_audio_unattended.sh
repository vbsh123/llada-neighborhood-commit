#!/usr/bin/env bash
# Setup + one experiment family. Designed to run under nohup on a new Vast machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
family="${1:-}"
case "$family" in prune|boost|combined) ;; *) echo 'Usage: bash scripts/vast_audio_unattended.sh [prune|boost|combined]'; exit 2;; esac
export SAMPLES="${SAMPLES:-100}"
export REFERENCE_DEVICE="${REFERENCE_DEVICE:-cuda}"
export VERIFY_SAMPLES="${VERIFY_SAMPLES:-2}"
export WHISPER_REVISION="${WHISPER_REVISION:-06f233fe06e710322aca913c1bc4249a0d71fce1}"
echo "Unattended setup, then $family on $SAMPLES questions"
env SKIP_INSTALL=0 bash scripts/vast_audio_drift.sh setup
export SKIP_INSTALL=1
# Preparation, disjointness/fingerprint verification and evaluation all run here.
exec bash scripts/vast_audio_independent.sh "$family"
