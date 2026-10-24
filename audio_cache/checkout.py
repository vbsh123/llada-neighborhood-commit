"""Fetch an immutable, unmodified official DIFFA checkout."""
import argparse
from pathlib import Path
import subprocess

COMMIT = "0a87594e110984fbaf00ebf5bfe5840c212e4438"
REPOSITORY = "https://github.com/NKU-HLT/DIFFA.git"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "external" / "DIFFA"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_checkout(root):
    root = Path(root).resolve()
    if not (root / ".git").is_dir():
        raise ValueError(f"Missing DIFFA checkout: {root}; run checkout.py first")
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("DIFFA root must be the checkout root")
    if git(root, "rev-parse", "HEAD") != COMMIT:
        raise ValueError(f"DIFFA must be at {COMMIT}; existing checkout left unchanged")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("DIFFA checkout has local changes; use a clean checkout")
    return root / "DIFFA-2"


def prepare_checkout(root):
    root = Path(root).resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", COMMIT], check=True)
    return verify_checkout(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffa-root", type=Path, default=DEFAULT_ROOT)
    print(prepare_checkout(parser.parse_args().diffa_root))
