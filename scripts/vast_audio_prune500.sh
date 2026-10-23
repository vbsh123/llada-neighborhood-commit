#!/usr/bin/env bash
# One GPU:500 unseen questions, baseline/audio/generic, one generation each.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then bash scripts/vast_audio_drift.sh setup; fi
audio_venv="${AUDIO_VENV:-$PWD/.venv-audio}"
source "$audio_venv/bin/activate"
count="${SAMPLES:-500}"
seed="${DATA_SEED:-20261024}"
run="${RUN_NAME:-audio_prune${count}_v23}"
manifest="${MANIFEST:-$PWD/data/audio_drift/gsm_heldout${count}_seed${seed}_v23.jsonl}"
revision=e94caa2154433a54cd430702666c454977df3083
exclude="$PWD/experiments/audio_drift/inspected_gsm_ids_v22.json"
[[ "$count" =~ ^[1-9][0-9]*$ && "$seed" =~ ^[0-9]+$ && "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid count/seed/run name'; exit 2; }
if [[ ! -f "$manifest" ]]; then
  python -u -m experiments.audio_drift.prepare_gsm --out "$manifest" --samples "$count" \
    --seed "$seed" --revision "$revision" --exclude-id-files "$exclude" --cohort heldout_v23
fi
python - "$manifest" "$count" "$seed" "$revision" "$exclude" <<'PY'
import json,sys,hashlib
from pathlib import Path
from experiments.audio_drift.inputs import read_manifest
p=Path(sys.argv[1]);count=int(sys.argv[2]);seed=int(sys.argv[3]);revision=sys.argv[4]
old=json.loads(Path(sys.argv[5]).read_text());rows=read_manifest(p)
ids=[r['id'] for r in rows];excluded=set(old['ids'])
assert len(rows)==count and len(set(ids))==count,'Unexpected sample count/duplicate IDs'
assert not set(ids)&excluded,'Held-out inputs overlap previously inspected questions'
assert all(r['source_revision']==revision and r['cohort']=='heldout_v23' for r in rows),'Wrong revision/cohort'
provenance=json.loads(p.with_suffix('.provenance.json').read_text())
assert provenance['seed']==seed and provenance['revision']==revision,'Wrong seed/revision'
assert provenance['min_seconds']==0 and provenance['max_seconds']==30,'Changed duration selection'
assert set(provenance['excluded_source_indices'])=={int(i.removeprefix('gsm8k-speech-')) for i in excluded},'Wrong exclusion pool'
content=[{k:r[k] for k in ('id','source_revision','prompt','target','metric','audio_sha256')} for r in rows]
sha=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
p.with_suffix('.fingerprint.json').write_text(json.dumps(dict(sha256=sha,samples=count,question_ids=ids,
    excluded_ids=sorted(excluded),seed=seed,revision=revision),indent=2)+'\n')
print(f'Validated {count} held-out questions; excluded all{len(excluded)} previously inspected IDs. INPUT_FINGERPRINT={sha}',flush=True)
PY
python -u -m experiments.audio_drift.run \
  --manifest "$manifest" --out "runs/$run" --samples "$count" \
  --length 512 --block-length 32 --steps 256 --layers all \
  --progressive-prune --lean-pruning --skip-dense-interventions --no-drift \
  --acoustic-start 16 --progressive-semantic-start 44 --progressive-interval 8 \
  --progressive-patience 2 --progressive-threshold 0.5 --progressive-fraction 0.25 \
  --benchmark-stopping eos --profile-samples 0 --verify-samples "${VERIFY_SAMPLES:-2}" \
  --reference-device cuda --cache-device model --cpu-threads "${CPU_THREADS:-4}" \
  --progress-every "${PROGRESS_EVERY:-32}" --seed 1729 \
  --whisper-revision 06f233fe06e710322aca913c1bc4249a0d71fce1
cp "${manifest%.*}.fingerprint.json" "runs/$run/input_fingerprint.json"
python -m experiments.audio_drift.reasoning_report --run "runs/$run"
python -m experiments.audio_drift.pruning_report --run "runs/$run"
python -m experiments.audio_drift.numeric_answer_report --run "runs/$run"
python -m experiments.audio_drift.lean_report --run "runs/$run"
echo "Complete: runs/$run"
