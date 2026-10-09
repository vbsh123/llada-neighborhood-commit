"""Set up and verify an unchanged, pinned checkout of the official SOAR repo."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from hooks import COMMIT, SOURCE_SHA256

REPOSITORY = 'https://github.com/duterscmy/SOAR.git'
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'external' / 'SOAR'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def verify_checkout(root):
    """Never reset an existing checkout or silently use edited upstream code."""
    root = Path(root).resolve()
    if not (root / '.git').exists():
        raise ValueError(f'Missing SOAR checkout: {root}. Run checkout.py first.')
    if Path(git(root, 'rev-parse', '--show-toplevel')).resolve() != root:
        raise ValueError('SOAR root must be the checkout root')
    if git(root, 'rev-parse', 'HEAD') != COMMIT:
        raise ValueError(f'SOAR checkout must be at {COMMIT}; existing checkout left unchanged')
    if git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('SOAR checkout has local changes; use a clean separate checkout')
    source = root / 'eval_llada8b' / 'generate.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Pinned SOAR generate.py hash differs')
    return source


def prepare_checkout(root):
    root = Path(root).resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', REPOSITORY, str(root)], check=True)
        subprocess.run(['git', '-C', str(root), 'checkout', '--detach', COMMIT], check=True)
    # Existing folders are checked, never overwritten/reset/pulled.
    return verify_checkout(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soar-root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(f'Pinned SOAR ready: {prepare_checkout(args.soar_root)}')
