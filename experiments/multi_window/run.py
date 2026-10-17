"""Vast-only GPU experiment; reuse the shared decoder and selective backend."""
import argparse
from pathlib import Path

from experiments.token_sleep import benchmark
from .schedule import WindowSchedule


def run(args):
    def factory(length, mode, **kwargs):
        return WindowSchedule(length, mode, radius=args.radius,region_gap=args.region_gap,
                              refresh_every=args.refresh_every)
    # Validate before dataset/model loading.
    factory(args.length, 'simulated_sleep')
    modes = ('multi_window',)
    if args.compare_active:
        modes = ('active_baseline', *modes)
    if args.include_original:
        modes = ('original', *modes)
    return benchmark.run(args, schedule_factory=factory, modes=modes,
                         experiment='persistent_mask_gap_multi_window_v2',
                         extra_sources=list(Path(__file__).parent.glob('*.py')))


def parser():
    # Same pinned model, dataset selection, prompt, scoring, resume, timing and
    # numerical validation as the sleep experiment. No entropy scheduling here.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=10)
    p.add_argument('--samples-from',type=Path)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--policy',choices=['top1','threshold'],default='top1')
    p.add_argument('--commit-threshold',type=float,default=.9)
    p.add_argument('--warmup-forwards',type=int,default=2)
    p.add_argument('--validation-steps',type=int,default=6)
    p.add_argument('--validation-tolerance',type=float,default=.02)
    p.add_argument('--allow-scaled-fp32-drift',action='store_true',
                   help='Existing explicit alternate numerical validation; actual execution stays BF16')
    p.add_argument('--include-original',action='store_true',help='Explicitly add vanilla generation')
    # Compatibility fields required by the shared runner. The supplied factory
    # ignores these entropy-sleep settings; users cannot set them in this CLI.
    p.set_defaults(original_only=False,sleep_only=False,diagnose_only=False,
                   sleep_confidence=.5,high_entropy=2.,high_sleep=2)
    p.add_argument('--radius', type=int, default=4)
    p.add_argument('--region-gap', type=int, default=4,
                   help='Consecutive MASKs required to separate two logical regions')
    p.add_argument('--refresh-every', type=int, default=4)
    p.add_argument('--compare-active', action='store_true',
                   help='Explicitly add full active computation; default treatment only')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    run(args)
