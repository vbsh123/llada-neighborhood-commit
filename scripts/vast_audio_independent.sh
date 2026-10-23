#!/usr/bin/env bash
# Frozen standalone evaluations on one disjoint question pool; no combinations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-prepare}"
case "$mode" in prepare|cache|prune|boost|cache-boost|all) ;; *) echo 'Usage: bash scripts/vast_audio_independent.sh [prepare|cache|prune|boost|cache-boost|all]'; exit 2;; esac
audio_venv="${AUDIO_VENV:-$PWD/.venv-audio}"
[[ -x "$audio_venv/bin/python" ]] || { echo 'Run bash scripts/vast_audio_drift.sh setup first'; exit 2; }
source "$audio_venv/bin/activate"
count="${SAMPLES:-50}"
seed="${DATA_SEED:-20261023}"
revision="${DATASET_REVISION:-e94caa2154433a54cd430702666c454977df3083}"
prefix="${RUN_PREFIX:-audio_independent}"
[[ "$count" =~ ^[1-9][0-9]*$ && "$seed" =~ ^[0-9]+$ && "$prefix" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid sample count/seed/run prefix'; exit 2; }
manifest="${MANIFEST:-$PWD/data/audio_drift/gsm_validation${count}_seed${seed}_v21.jsonl}"
read -r -a excludes <<< "${EXCLUDE_MANIFESTS:-}"
id_file="$PWD/experiments/audio_drift/development_gsm_ids_v20.json"
exclusion_args=(--exclude-id-files "$id_file")
if (( ${#excludes[@]} )); then exclusion_args+=(--exclude-manifests "${excludes[@]}"); fi
for path in "${excludes[@]}"; do [[ -f "$path" ]] || { echo "Missing exclusion manifest: $path"; exit 2; }; done
if [[ ! -f "$manifest" ]]; then
  python -u -m experiments.audio_drift.prepare_gsm --out "$manifest" --samples "$count" \
    --revision "$revision" --seed "$seed" "${exclusion_args[@]}"
fi
python - "$manifest" "$count" "$seed" "$revision" "$id_file" "${excludes[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
from experiments.audio_drift.inputs import read_manifest
path=Path(sys.argv[1]);count,seed=int(sys.argv[2]),int(sys.argv[3]);revision=sys.argv[4]
rows=read_manifest(path);old=[r for p in sys.argv[6:] for r in read_manifest(p)]
ids=json.loads(Path(sys.argv[5]).read_text())
old += [dict(id=i,source_revision=ids['source_revision']) for i in ids['ids']]
excluded={r['id'] for r in old};new={r['id'] for r in rows}
assert len(rows)==count and len(new)==count,'Unexpected validation sample count'
assert not new&excluded,'Validation questions overlap prior experiments'
assert all(r.get('source_revision')==revision for r in rows+old),'Dataset revision mismatch'
assert all(r.get('cohort')=='validation_v21' for r in rows),'Use a dedicated exclusion-only validation manifest'
provenance=json.loads(path.with_suffix('.provenance.json').read_text())
assert provenance['seed']==seed and provenance['revision']==revision,'Manifest provenance mismatch'
assert set(provenance['excluded_source_indices'])=={int(i.removeprefix('gsm8k-speech-')) for i in excluded},'Exclusion set changed'
assert provenance['min_seconds']==0 and provenance['max_seconds']==30,'Duration distribution changed'
print(f'Validated {count} new questions; excluded {len(excluded)} prior questions; seed={seed}',flush=True)
content=[{k:r[k] for k in ('id','source_revision','prompt','target','metric','audio_sha256')} for r in rows]
fingerprint=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
path.with_suffix('.fingerprint.json').write_text(json.dumps(dict(sha256=fingerprint,samples=count,seed=seed,revision=revision),indent=2)+'\n')
print(f'INPUT_FINGERPRINT={fingerprint}',flush=True)
PY
if [[ "$mode" == prepare ]]; then echo "Manifest: $manifest"; exit 0; fi
run_family() {
  local family="$1"
  local launcher
  case "$family" in cache) launcher=gsm-cache;; prune) launcher=gsm-progressive-prune;; boost) launcher=gsm-prompt-boost;; esac
  local run="${prefix}_${family}${count}_v21"
  local replay="${REPLAY_SAMPLES:-2}"
  if (( replay > count )); then replay="$count"; fi
  echo "Starting standalone $family on $count new questions: runs/$run"
  env MANIFEST="$manifest" SAMPLES="$count" RUN_NAME="$run" \
    SKIP_INSTALL="${SKIP_INSTALL:-1}" WHISPER_REVISION="${WHISPER_REVISION:-06f233fe06e710322aca913c1bc4249a0d71fce1}" \
    REPLAY_SAMPLES="$replay" BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-2}" REPLAY_REPEATS="${REPLAY_REPEATS:-2}" \
    ACOUSTIC_START=16 SEMANTIC_START=44 PRUNE_INTERVAL=8 PRUNE_PATIENCE=2 PRUNE_THRESHOLD=0.5 PRUNE_ELIGIBLE_FRACTION=0.25 \
    PRIORITY_STRENGTHS="0.05 0.1" PRIORITY_MIDPOINT=44 PRIORITY_END=88 \
    bash scripts/vast_audio_drift.sh "$launcher"
  python -m experiments.audio_drift.numeric_answer_report --run "runs/$run"
  cp "${manifest%.*}.fingerprint.json" "runs/$run/input_fingerprint.json"
}
if [[ "$mode" == all || "$mode" == cache-boost ]]; then
  families=(cache prune boost)
  if [[ "$mode" == cache-boost ]]; then families=(cache boost); fi
  for family in "${families[@]}"; do run_family "$family"; done
else
  run_family "$mode"
fi
