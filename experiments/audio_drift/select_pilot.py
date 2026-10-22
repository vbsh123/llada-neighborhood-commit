"""Select a small duration-only pilot from an existing pinned manifest."""
import argparse
import json
from pathlib import Path
from .inputs import read_manifest


def select(rows,count,min_seconds):
    chosen=[r for r in rows if r.get('audio_seconds',0)>=min_seconds][:count]
    if count<1 or min_seconds<0 or len(chosen)!=count:
        raise ValueError('Not enough duration-eligible clips for this pilot')
    return chosen


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=2);p.add_argument('--min-seconds',type=float,default=15)
    args=p.parse_args();rows=select(read_manifest(args.source),args.samples,args.min_seconds)
    content=''.join(json.dumps(r)+'\n' for r in rows)
    if args.out.exists() and args.out.read_text()!=content:
        raise ValueError('Existing pilot manifest differs; use a new output path')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(content)
    args.out.with_suffix('.provenance.json').write_text(json.dumps(dict(source=str(args.source.resolve()),
        samples=args.samples,min_seconds=args.min_seconds,selection='first duration-eligible rows; no correctness filtering'),indent=2)+'\n')
    print(args.out,flush=True)
