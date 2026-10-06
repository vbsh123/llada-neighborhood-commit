"""Read completed traces from a directory or ZIP without extracting them."""
import gzip
import io
import json
from pathlib import Path
import zipfile


class Source:
    def __init__(self, path):
        self.path=Path(path)
        self.archive=zipfile.ZipFile(path) if self.path.is_file() else None
        self.names=sorted(n for n in self.archive.namelist() if n.endswith('/trace.jsonl.gz')) if self.archive else sorted(str(p) for p in self.path.rglob('trace.jsonl.gz'))
        if not self.names:
            self.close();raise ValueError(f'No completed traces in {path}')

    def close(self):
        if self.archive:self.archive.close()

    def raw(self,name):
        return self.archive.open(name) if self.archive else open(name,'rb')

    def result(self,name):
        with self.raw(name.replace('trace.jsonl.gz','result.json')) as handle:return json.load(handle)

    def records(self,name):
        with self.raw(name) as raw,gzip.GzipFile(fileobj=raw) as gz,io.TextIOWrapper(gz,encoding='utf-8') as handle:
            for line in handle:yield json.loads(line)
