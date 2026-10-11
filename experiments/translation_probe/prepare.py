"""Reproduce the checked-in ten-example selection from the original FLORES archive."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tarfile

URL='https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz'
SHA256='b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6'
HERE=Path(__file__).resolve().parent


def prepare(archive, out=HERE/'inputs', sentences_per_input=1):
    if sentences_per_input not in (1,4):
        raise ValueError('Supported input sizes: one or four sentences')
    archive=Path(archive);out=Path(out)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Archive differs from the pinned original FLORES-200 release')
    files={lang:f'./flores200_dataset/dev/{lang}.dev' for lang in ('zho_Hans','eng_Latn')}
    with tarfile.open(archive,'r:gz') as handle:
        # Read only these members; never extract paths from a tar archive.
        texts={lang:handle.extractfile(name).read().decode('utf-8').splitlines() for lang,name in files.items()}
    if len(texts['zho_Hans'])!=997 or len(texts['eng_Latn'])!=997:
        raise ValueError('Expected 997 aligned dev sentences')
    indices=list(range(997));random.Random(1729).shuffle(indices);indices=indices[:10*sentences_per_input]
    rows=[{'id':f'{i+1:05d}','flores_id':i+1,'dataset_index':i,'source_language':'zho_Hans',
           'target_language':'eng_Latn','source_text':texts['zho_Hans'][i],'reference':texts['eng_Latn'][i]} for i in indices]
    if sentences_per_input==4:
        rows=[{'id':f'{j+1:05d}','source_language':'zho_Hans','target_language':'eng_Latn',
               'source_text':'\n'.join(r['source_text'] for r in rows[j*4:j*4+4]),
               'reference':'\n'.join(r['reference'] for r in rows[j*4:j*4+4]),
               'flores_ids':[r['flores_id'] for r in rows[j*4:j*4+4]],
               'sentence_pairs':rows[j*4:j*4+4]} for j in range(10)]
    out.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (out/'flores_zh_en_dev_10.jsonl').write_text(content,encoding='utf-8')
    info={'dataset':'FLORES-200 original July 2022 release','split':'dev',
          'official_documentation':'https://github.com/facebookresearch/flores/tree/main/flores200',
          'archive_url':URL,'archive_sha256':SHA256,'archive_members':files,
          'license':'CC-BY-SA-4.0','license_url':'https://creativecommons.org/licenses/by-sa/4.0/',
          'attribution':'FLORES-200 / NLLB Team, Meta AI; No Language Left Behind (2022).',
          'selection':'Shuffle all 997 aligned dev row indices with Python random.Random(1729); first 10, without length/quality filtering.',
          'seed':1729,'flores_ids':[i+1 for i in indices],
          'selected_inputs_sha256':hashlib.sha256(content.encode()).hexdigest()}
    if sentences_per_input==4:
        info['selection']='Shuffle all 997 aligned dev row indices with Python random.Random(1729); first 40, grouped in fours in shuffled order, without length/quality filtering.'
        info['sentences_per_input']=4
        info['grouping']='Unrelated sentences joined with newlines in both languages; no numbering or extra instructions. References are never passed to the model. Not a coherent paragraph dataset; token lengths are not constrained.'
    (out/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
    print(f'Prepared {len(rows)} aligned examples in {out}',flush=True)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE/'inputs')
    parser.add_argument('--sentences-per-input',type=int,choices=[1,4],default=1)
    args=parser.parse_args();prepare(args.archive,args.out,args.sentences_per_input)
