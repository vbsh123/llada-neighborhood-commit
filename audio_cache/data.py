"""Restore a frozen source-row cohort and verify every original WAV hash."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf


COHORT = Path(__file__).with_name('cohort.json')


def fingerprint(rows):
    keys=('id','source_revision','prompt','target','metric','audio_sha256')
    return hashlib.sha256(json.dumps([{k:r[k] for k in keys} for r in rows],
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_manifest(path,reference_path=None):
    path=Path(path).resolve()
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected=json.loads(Path(reference_path or COHORT).read_text())
    if (len(rows)!=len(expected['rows']) or len({r['id'] for r in rows})!=len(rows)
            or fingerprint(rows)!=expected['sha256']):
        raise ValueError('Manifest differs from the frozen question cohort')
    for row in rows:
        audio=(path.parent/row['audio']).resolve()
        if hashlib.sha256(audio.read_bytes()).hexdigest()!=row['audio_sha256']:
            raise ValueError('Audio hash mismatch: '+row['id'])
        row['audio']=str(audio)
    return rows


def prepare_manifest(out,reference_path=None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    out=Path(out)
    if out.exists():
        rows=read_manifest(out,reference_path)
        print(f'Validated exact {len(rows)} questions: {out}',flush=True)
        return
    reference=json.loads(Path(reference_path or COHORT).read_text())
    rows=reference['rows']
    out.parent.mkdir(parents=True,exist_ok=True)
    audio_dir=out.parent/(out.stem+'_audio')
    audio_dir.mkdir(exist_ok=True)
    by_source={}
    prepared={}
    for i,row in enumerate(rows):
        path=audio_dir/f'{i:05d}.wav'
        original=Path(row['original_audio']) if row.get('original_audio') else None
        if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==row['audio_sha256']):
            if original and original.is_file() and hashlib.sha256(original.read_bytes()).hexdigest()==row['audio_sha256']:
                if original.resolve()!=path.resolve():shutil.copyfile(original,path)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==row['audio_sha256']:
            prepared[i]={**row,'audio':str(path.relative_to(out.parent))}
            continue
        by_source.setdefault(row['source_file'],{})[row['source_row']]=(i,row)
    for source,wanted in by_source.items():
        local=hf_hub_download(rows[0]['source_dataset'],source,repo_type='dataset',revision=rows[0]['source_revision'])
        with open(local,'rb') as handle, pq.ParquetFile(handle) as parquet:
            offset=0
            for batch in parquet.iter_batches(batch_size=4,use_threads=False):
                for item in batch.to_pylist():
                    selected=wanted.get(offset)
                    offset+=1
                    if selected is None:
                        continue
                    index,row=selected
                    audio,rate=sf.read(io.BytesIO(item['audio']['bytes']),dtype='float32')
                    path=audio_dir/f'{index:05d}.wav'
                    sf.write(path,audio,rate,subtype='PCM_16')
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['audio_sha256']:
                        raise ValueError('Downloaded row does not reproduce original WAV: '+row['id'])
                    prepared[index]={**row,'audio':str(path.relative_to(out.parent))}
                    print(f'Restored {len(prepared)}/{len(rows)}: {row["id"]}',flush=True)
                if offset>max(wanted):
                    break
    if len(prepared)!=len(rows):
        raise ValueError('Some original source rows were not found')
    temporary=out.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(prepared[i])+'\n' for i in range(len(rows))))
    temporary.replace(out)
    read_manifest(out,reference_path)
    print(f'Exact {len(rows)} questions verified: {out}',flush=True)


def load_audio(path):
    import torch
    import torchaudio
    audio,rate=sf.read(path,dtype='float32',always_2d=True)
    if not audio.size or not np.isfinite(audio).all() or len(audio)/rate>30:
        raise ValueError('Invalid or over30-second audio')
    wave=torch.from_numpy(audio.mean(axis=1))
    if rate!=16000:
        wave=torchaudio.functional.resample(wave,rate,16000)
    return wave.numpy(),len(audio)/rate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('data/development50.jsonl'))
    prepare_manifest(parser.parse_args().out)
