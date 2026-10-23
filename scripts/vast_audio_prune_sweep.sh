#!/usr/bin/env bash
# Development50: five settings, audio/generic each; one baseline per question.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then bash scripts/vast_audio_drift.sh setup; fi
audio_venv="${AUDIO_VENV:-$PWD/.venv-audio}"
source "$audio_venv/bin/activate"
pool="${DEV_POOL_MANIFEST:-$PWD/data/audio_drift/gsm_validation100_seed20261023_v21.jsonl}"
manifest="$PWD/data/audio_drift/gsm_development50_v24.jsonl"
run="${RUN_NAME:-audio_prunesweep50_v24}"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run name'; exit 2; }
env MANIFEST="$pool" SAMPLES=50 POOL_SAMPLES=100 DATA_SEED=20261023 \
  DATASET_REVISION=e94caa2154433a54cd430702666c454977df3083 EXCLUDE_MANIFESTS= \
  bash scripts/vast_audio_independent.sh prepare
python - "$pool" "$manifest" <<'PY'
import json,sys,hashlib
from pathlib import Path
from experiments.audio_drift.inputs import read_manifest
pool=Path(sys.argv[1]);manifest=Path(sys.argv[2]);rows=read_manifest(pool,50)
content=[{k:r[k] for k in ('id','source_revision','prompt','target','metric','audio_sha256')} for r in rows]
sha=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert sha=='8ed78386500a79daecdf9b9be573efe2c2a4001a5c667b23c58555a6e03205f2','Not the exact previous development50 inputs'
for r in rows:r['cohort']='development_v24'
text=''.join(json.dumps(r)+'\n' for r in rows)
if manifest.exists():assert manifest.read_text()==text,'Development manifest changed'
else:manifest.write_text(text)
manifest.with_suffix('.fingerprint.json').write_text(json.dumps(dict(sha256=sha,samples=50,
    question_ids=[r['id'] for r in rows],cohort='development_v24',source_pool=str(pool)),indent=2)+'\n')
print('Confirmed exact50 previously inspected questions. Five configurations; audio/generic per configuration.',flush=True)
PY
python -u -m experiments.audio_drift.run \
  --manifest "$manifest" --out "runs/$run" --samples 50 \
  --length 512 --block-length 32 --steps 256 --layers all \
  --progressive-prune --lean-pruning --pruning-sweep --shuffle-policy-order \
  --skip-dense-interventions --no-drift --profile-samples 0 --benchmark-stopping eos \
  --acoustic-start 16 --progressive-semantic-start 44 --progressive-interval 8 --progressive-patience 2 \
  --verify-samples "${VERIFY_SAMPLES:-2}" --progress-every 0 --seed 1729 \
  --reference-device cuda --cache-device model --cpu-threads "${CPU_THREADS:-4}" \
  --whisper-revision 06f233fe06e710322aca913c1bc4249a0d71fce1
cp "${manifest%.*}.fingerprint.json" "runs/$run/input_fingerprint.json"
python -m experiments.audio_drift.numeric_answer_report --run "runs/$run"
python -m experiments.audio_drift.lean_report --run "runs/$run"
python -m experiments.audio_drift.pruning_sweep --run "runs/$run"
echo "Complete: runs/$run"
