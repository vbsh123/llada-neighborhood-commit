#!/usr/bin/env bash
# Run from the repo checkout on Vast. Uses its own environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-smoke}"
case "$mode" in
  setup|smoke|pilot|long-pilot|control-pilot|cosine-pilot|gsm-pilot|gsm-sweep|gsm-depth|gsm-night|gsm-diagnose) ;;
  *) echo 'Usage: bash scripts/vast_audio_drift.sh [setup|smoke|pilot|long-pilot|control-pilot|cosine-pilot|gsm-pilot|gsm-sweep|gsm-depth|gsm-night|gsm-diagnose]'; exit 2 ;;
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
if [[ "$mode" == gsm-night || "$mode" == gsm-diagnose ]]; then
  manifest="${MANIFEST:-$PWD/data/audio_drift/gsm8k_speech30_v10.jsonl}"
  base_manifest="${BASE_MANIFEST:-$PWD/data/audio_drift/gsm8k_speech10_v7.jsonl}"
  if [[ ! -f "$manifest" && -z "${MANIFEST:-}" ]]; then
    if [[ ! -f "$base_manifest" ]]; then
      python -m experiments.audio_drift.prepare_gsm --out "$base_manifest" --samples 10 --seed 1729
    fi
    python -m experiments.audio_drift.prepare_gsm --out "$manifest" --samples 30 \
      --base-manifest "$base_manifest" --seed "${DATA_SEED:-20261020}"
  fi
  default_count=30
  if [[ "$mode" == gsm-diagnose ]]; then default_count=6; fi
  count="${SAMPLES:-$default_count}"
  length="${LENGTH:-512}"
  block="${BLOCK_LENGTH:-32}"
elif [[ "$mode" == gsm-pilot || "$mode" == gsm-sweep || "$mode" == gsm-depth ]]; then
  manifest="${MANIFEST:-$PWD/data/audio_drift/gsm8k_speech10_v7.jsonl}"
  if [[ ! -f "$manifest" && -z "${MANIFEST:-}" ]]; then
    python -m experiments.audio_drift.prepare_gsm --out "$manifest" --samples 10 --seed "${SEED:-1729}"
  fi
  count="${SAMPLES:-10}"
  gsm_capacity=256
  if [[ "$mode" == gsm-sweep || "$mode" == gsm-depth ]]; then gsm_capacity=512; fi
  length="${LENGTH:-$gsm_capacity}"
  block="${BLOCK_LENGTH:-32}"
elif [[ "$mode" == long-pilot || "$mode" == control-pilot || "$mode" == cosine-pilot ]]; then
  manifest="${MANIFEST:-$PWD/data/audio_drift/librispeech_duration20_v4.jsonl}"
  if [[ ! -f "$manifest" && -z "${MANIFEST:-}" ]]; then
    python -m experiments.audio_drift.prepare --out "$manifest" --samples 20 \
      --dataset openslr/librispeech_asr --config clean --split test --reader local-parquet \
      --shuffle-seed "${SEED:-1729}" --max-per-speaker 2 \
      --min-seconds 10 --max-seconds 30 --duration-bins 10:20:10 20:30:10
  fi
  default_count=20
  if [[ "$mode" == control-pilot || "$mode" == cosine-pilot ]]; then default_count=2; fi
  count="${SAMPLES:-$default_count}"
  length="${LENGTH:-256}"
  block="${BLOCK_LENGTH:-32}"
elif [[ "$mode" == smoke ]]; then
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
if [[ "$mode" == gsm-diagnose ]]; then
  default_refresh_samples=2
  if (( count < default_refresh_samples )); then default_refresh_samples="$count"; fi
  extra+=(--interventions freeze --freeze-groups prefix --refresh-intervals 1 8
          --triple-refresh 8:8:16 16:8:8 16:8:16 32:8:32 0:8:0
          --condition-names freeze_prefix_r1 freeze_prefix_r8 freeze_triple_a8_p8_s16
          freeze_triple_a16_p8_s8 freeze_triple_a16_p8_s16 freeze_triple_a32_p8_s32 freeze_triple_a0_p8_s0
          --skip-dense-interventions --no-drift --shuffle-policy-order
          --benchmark-samples "$count" --benchmark-repeats "${BENCHMARK_REPEATS:-2}" --benchmark-stopping eos
          --replay-samples "$count" --replay-steps "${REPLAY_STEPS:-128}" --replay-repeats "${REPLAY_REPEATS:-2}"
          --diagnostic-samples "${DIAGNOSTIC_SAMPLES:-$count}" --diagnostic-layers "${DIAGNOSTIC_LAYERS:-all}"
          --diagnostic-queries "${DIAGNOSTIC_QUERIES:-4}" --diagnostic-tokens
          --refresh-profile-samples "${REFRESH_PROFILE_SAMPLES:-$default_refresh_samples}" --refresh-profile-steps "${REPLAY_STEPS:-128}"
          --refresh-profile-repeats "${REFRESH_PROFILE_REPEATS:-3}")
  if [[ "${REFRESH_TRACE:-1}" == 1 ]]; then extra+=(--refresh-trace); fi
elif [[ "$mode" == gsm-night ]]; then
  extra+=(--interventions freeze --freeze-groups prefix --refresh-intervals 1 8
          --triple-refresh 16:8:16 16:8:8 16:8:4 8:8:4 16:16:4
          --condition-names freeze_prefix_r1 freeze_prefix_r8 freeze_triple_a16_p8_s16
          freeze_triple_a16_p8_s8 freeze_triple_a16_p8_s4 freeze_triple_a8_p8_s4 freeze_triple_a16_p16_s4
          --depth-multipliers 8 4 2 1 --skip-dense-interventions --no-drift --shuffle-policy-order
          --benchmark-samples "${BENCHMARK_SAMPLES:-$count}" --benchmark-repeats "${BENCHMARK_REPEATS:-2}"
          --benchmark-stopping eos
          --replay-samples "${REPLAY_SAMPLES:-10}" --replay-steps "${REPLAY_STEPS:-64}"
          --replay-repeats "${REPLAY_REPEATS:-2}")
elif [[ "$mode" == gsm-depth ]]; then
  : # Baseline only; no generation policies, benchmarks or pairwise vectors.
elif [[ "$mode" == gsm-sweep ]]; then
  extra+=(--interventions freeze --freeze-groups prefix --refresh-intervals 1 8 16
          --triple-refresh 16:8:8 32:8:8 0:8:8 16:8:16
          --condition-names freeze_prefix_r1 freeze_prefix_r8 freeze_prefix_r16
          freeze_triple_a16_p8_s8 freeze_triple_a32_p8_s8 freeze_triple_a0_p8_s8
          freeze_triple_a16_p8_s16
          --skip-dense-interventions
          --benchmark-samples "${BENCHMARK_SAMPLES:-$count}" --benchmark-repeats "${BENCHMARK_REPEATS:-2}"
          --benchmark-stopping eos
          --replay-samples "${REPLAY_SAMPLES:-2}" --replay-steps "${REPLAY_STEPS:-64}"
          --replay-repeats "${REPLAY_REPEATS:-2}")
elif [[ "$mode" == gsm-pilot ]]; then
  extra+=(--interventions freeze --freeze-groups prefix --refresh-intervals 1 4 8
          --triple-refresh 8:4:4 16:4:4 8:8:4
          --condition-names freeze_prefix_r1 freeze_prefix_r4 freeze_prefix_r8
          freeze_triple_a8_p4_s4 freeze_triple_a16_p4_s4 freeze_triple_a8_p8_s4
          --skip-dense-interventions
          --benchmark-samples "${BENCHMARK_SAMPLES:-$count}" --benchmark-repeats "${BENCHMARK_REPEATS:-2}"
          --benchmark-stopping "${BENCHMARK_STOPPING:-eos}")
elif [[ "$mode" == cosine-pilot ]]; then
  extra+=(--pairwise-count "${PAIRWISE_COUNT:-64}" --pairwise-stride "${PAIRWISE_STRIDE:-1}"
          --pairwise-layers "${PAIRWISE_LAYERS:-7,15,23,31}" --pairwise-device "${PAIRWISE_DEVICE:-cuda}")
elif [[ "$mode" == control-pilot ]]; then
  extra+=(--interventions freeze --freeze-groups acoustic semantic both prompt audio_matched prefix
          --refresh-intervals 1 4 8 --mixed-refresh 8:2 2:8
          --condition-names freeze_acoustic freeze_semantic freeze freeze_both_r1 freeze_both_r4
          freeze_both_r8 freeze_mixed_a8_s2 freeze_mixed_a2_s8 freeze_prompt freeze_prompt_r1
          freeze_prompt_r4 freeze_audio_matched_r4 freeze_prefix_r4
          --benchmark-samples "${BENCHMARK_SAMPLES:-2}" --benchmark-repeats "${BENCHMARK_REPEATS:-2}"
          --benchmark-stopping "${BENCHMARK_STOPPING:-eos}"
          --benchmark-conditions freeze_both_r1 freeze_prompt_r4 freeze_audio_matched_r4
          freeze_both_r4 freeze_prefix_r4 freeze_acoustic freeze_mixed_a8_s2)
elif [[ "$mode" == long-pilot ]]; then
  extra+=(--interventions freeze --freeze-groups acoustic semantic both
          --refresh-intervals 1 4 8 --mixed-refresh 8:2 2:8
          --condition-names freeze_acoustic freeze_semantic freeze freeze_both_r1
          freeze_both_r4 freeze_both_r8 freeze_mixed_a8_s2 freeze_mixed_a2_s8)
elif [[ "${GROUP_STUDY:-0}" == 1 ]]; then
  read -r -a refresh_intervals <<< "${REFRESH_INTERVALS:-1 2 4 8}"
  read -r -a mixed_refresh <<< "${MIXED_REFRESH:-8:2 2:8}"
  extra+=(--interventions freeze remove text_only --freeze-groups acoustic semantic both
          --refresh-intervals "${refresh_intervals[@]}" --mixed-refresh "${mixed_refresh[@]}")
elif [[ "${INTERVENTIONS:-0}" == 1 ]]; then
  extra+=(--interventions freeze remove text_only)
fi
default_fractions="0 .25 .5 .75"
default_profiles=0
default_layers=all
default_steps="$length"
if [[ "$mode" == long-pilot || "$mode" == control-pilot || "$mode" == cosine-pilot ]]; then
  default_fractions=0; default_profiles=2; default_layers=0,7,15,23,31; default_steps=128
fi
if [[ "$mode" == control-pilot ]]; then default_profiles=0; fi
if [[ "$mode" == gsm-pilot ]]; then
  default_fractions=0; default_profiles=0; default_layers=all; default_steps=128
fi
if [[ "$mode" == gsm-sweep || "$mode" == gsm-depth || "$mode" == gsm-night || "$mode" == gsm-diagnose ]]; then
  default_fractions=0; default_profiles=0; default_layers=all
  default_steps=$((length / 2))
fi
if [[ "$mode" == cosine-pilot ]]; then default_profiles=0; default_layers=all; fi
read -r -a branch_fractions <<< "${BRANCH_FRACTIONS:-$default_fractions}"
extra+=(--profile-samples "${PROFILE_SAMPLES:-$default_profiles}")
if [[ "${TOKEN_METRICS:-0}" == 1 ]]; then extra+=(--token-metrics); fi
python -u -m experiments.audio_drift.run \
  --diffa-root "$diffa_checkout" --manifest "$manifest" --out "runs/$experiment_name" \
  --samples "$count" --length "$length" --block-length "$block" --steps "${STEPS:-$default_steps}" \
  --layers "${LAYERS:-$default_layers}" --reference-device "${REFERENCE_DEVICE:-cpu}" \
  --cache-device "${CACHE_DEVICE:-model}" \
  --cpu-threads "${CPU_THREADS:-4}" \
  --verify-samples "${VERIFY_SAMPLES:-1}" --seed "${SEED:-1729}" \
  --progress-every "${PROGRESS_EVERY:-8}" --branch-fractions "${branch_fractions[@]}" "${extra[@]}"
if [[ "$mode" == gsm-pilot || "$mode" == gsm-sweep || "$mode" == gsm-night || "$mode" == gsm-diagnose ]]; then
  python -m experiments.audio_drift.reasoning_report --run "runs/$experiment_name"
fi
if [[ "$mode" == gsm-diagnose ]]; then
  python -m experiments.audio_drift.diagnostic_report --run "runs/$experiment_name"
fi
if [[ "$mode" != gsm-night && "$mode" != gsm-diagnose ]]; then
python -m experiments.audio_drift.analyze --run "runs/$experiment_name"
python -m experiments.audio_drift.stability --run "runs/$experiment_name" --statistic median
python -m experiments.audio_drift.stability --run "runs/$experiment_name" --statistic p90
fi
if [[ "$mode" == cosine-pilot || "$mode" == gsm-pilot || "$mode" == gsm-sweep || "$mode" == gsm-depth ]]; then
  python -m experiments.audio_drift.paper_plots --run "runs/$experiment_name"
fi
python -m pip freeze > "runs/$experiment_name/packages.txt"
nvidia-smi > "runs/$experiment_name/nvidia-smi.txt"
echo "Complete: runs/$experiment_name/summary.json"
