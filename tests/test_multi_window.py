"""CPU fixtures only: scheduling, real sparse math, shared validation/decoder."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from experiments.multi_window.schedule import WindowSchedule, merge_windows, region_bounds
from experiments.multi_window.run import parser, run
from experiments.token_sleep.benchmark import validate, decode, evaluate
from test_real_token_sleep import Model, Tokenizer


class ScheduleTests(unittest.TestCase):
    def test_merge_and_invalid_geometry(self):
        self.assertEqual(merge_windows([(8,10),(1,3),(4,6),(10,12)]),[(1,6),(8,12)])
        for kwargs in (dict(radius=-1),dict(refresh_every=0)):
            with self.assertRaises(ValueError): WindowSchedule(20,'simulated_sleep',**kwargs)

    def test_exact_mask_gap_and_interior(self):
        self.assertEqual(region_bounds([2,6],4),[(2,6)])  # three MASKs
        self.assertEqual(region_bounds([2,7],4),[(2,2),(7,7)])  # four MASKs
        with self.assertRaises(ValueError): region_bounds([2],0)
        schedule=WindowSchedule(20,'simulated_sleep',radius=0,region_gap=4)
        state=[0]*20; state[2]=state[6]=1
        sleepers=schedule.prepare(1,state,0)
        self.assertEqual(schedule.windows,[(2,6)])
        for p in range(2,7): self.assertNotIn(p,sleepers)
        self.assertIn(7,sleepers)

    def test_padding_overlap_does_not_merge_regions(self):
        schedule=WindowSchedule(30,'simulated_sleep',radius=4,region_gap=4)
        state=[0]*30
        schedule.prepare(0,state,0); schedule.finish(0,[],[5,10])
        self.assertEqual(schedule.regions,[(5,5),(10,10)])
        self.assertEqual(schedule.windows,[(1,14)])
        self.assertEqual(schedule.stats['regions_opened'],2)
        self.assertEqual(schedule.stats['windows_opened'],1)
        state[5]=state[10]=1
        schedule.prepare(1,state,0); schedule.finish(1,[],[7])
        self.assertEqual(schedule.regions,[(5,10)])
        self.assertEqual(schedule.stats['regions_merged'],1)
        self.assertEqual(schedule.stats['windows_merged'],0)

    def test_inactive_and_completed_regions_never_retire(self):
        schedule=WindowSchedule(30,'simulated_sleep',radius=1,region_gap=4)
        state=[0]*30
        for p in range(4,9): state[p]=1
        state[22]=1
        for step in [1,2,3,5,6,7,9,1001]:
            sleepers=schedule.prepare(step,state,0)
            self.assertEqual(schedule.regions,[(4,8),(22,22)])
            for p in list(range(3,10))+list(range(21,24)):
                self.assertNotIn(p,sleepers)
            self.assertIn(15,sleepers)
            self.assertFalse(set(sleepers) & {p for p,t in enumerate(state) if t})
        self.assertEqual(schedule.stats['windows_retired'],0)
        self.assertEqual(schedule.stats['sleeping_filled_position_forwards'],0)

    def test_refresh_discovery_and_exhausted_padding_fallback(self):
        schedule=WindowSchedule(30,'simulated_sleep',radius=1,region_gap=4,refresh_every=4)
        state=[0]*30
        self.assertEqual(schedule.prepare(0,state,0),[])
        schedule.finish(0,[],[5]); state[5]=1
        self.assertIn(20,schedule.prepare(1,state,0))
        self.assertEqual(schedule.prepare(4,state,0),[])
        schedule.finish(4,[],[20]); state[20]=1
        self.assertEqual(schedule.regions,[(5,5),(20,20)])
        self.assertEqual(schedule.windows,[(4,6),(19,21)])
        schedule.radius=0
        # With no interior MASK or padding candidate, refresh without retirement.
        self.assertEqual(schedule.prepare(5,state,0),[])
        self.assertTrue(schedule.fallback)
        self.assertEqual(schedule.windows,[(5,5),(20,20)])

    def test_baseline_refresh_one_and_edges(self):
        for mode,refresh in [('baseline',8),('simulated_sleep',1)]:
            schedule=WindowSchedule(12,mode,radius=2,refresh_every=refresh)
            state=[0]*12
            for step in range(3):
                self.assertEqual(schedule.prepare(step,state,0),[])
                schedule.finish(step,[],[0,11]); state[0]=state[11]=1
            if mode!='baseline': self.assertEqual(schedule.windows,[(0,2),(9,11)])

    def test_default_runs_no_baseline(self):
        args=parser().parse_args(['--out','fixture'])
        with patch('experiments.multi_window.run.benchmark.run') as runner:
            run(args)
        self.assertEqual(runner.call_args.kwargs['modes'],('multi_window',))


class IntegrationTests(unittest.TestCase):
    def test_window_validation_and_decode_on_toy_model(self):
        torch.manual_seed(57)
        model=Model().eval()
        config=dict(length=12,mask_id=0,sleep_confidence=.5,high_entropy=2.,high_sleep=2,
                    policy='top1',commit_threshold=.9,validation_steps=6,
                    validation_tolerance=.02,warmup_forwards=0)
        def factory(length,mode,**kwargs):
            return WindowSchedule(length,mode,radius=1,refresh_every=4)
        blocks=lambda m:list(m.model.transformer.blocks)
        sample=dict(id='fixture',question='fixture',answer='#### 1')
        with patch('experiments.token_sleep.selective.checked_blocks',blocks), \
             patch('experiments.token_sleep.benchmark.checked_blocks',blocks):
            report=validate(model,Tokenizer(),sample,config,factory)
            self.assertTrue(report['passed'])
            self.assertGreater(report['tested_sleep_forwards'],0)
            with tempfile.TemporaryDirectory() as tmp, \
                 patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), \
                 patch('torch.cuda.max_memory_allocated',return_value=0):
                out=Path(tmp)
                for mode in ('active_baseline','multi_window'):
                    directory=out/mode/'samples'/'fixture'; directory.mkdir(parents=True)
                    result=decode(model,Tokenizer(),sample,config,mode,directory,factory)
                    self.assertEqual(result['steps'],12)
                    self.assertNotIn(0,result['final_ids'])
                    self.assertEqual(result['active_transformer_rows'],
                        result['steps']*14-result['sleep_position_forwards']-
                        result['window_stats']['sleeping_filled_position_forwards'])
                    with gzip.open(directory/'trace.jsonl.gz','rt') as f:
                        steps=[r for line in f if (r:=json.loads(line))['type']=='step']
                    for step in steps:
                        self.assertFalse(set(step['commit_positions']) & set(step['sleep_positions']))
                        for p in step['commit_positions']:
                            self.assertTrue(step['window_schedule']['global_refresh'] or any(
                                a<=p<=b for a,b in step['window_schedule']['windows_before']))
                summary=evaluate(out,[sample],('active_baseline','multi_window'))
                self.assertIn('active_baseline_vs_multi_window',summary['comparisons'])
                self.assertLess(summary['modes']['multi_window']['active_transformer_rows'],
                                summary['modes']['active_baseline']['active_transformer_rows'])


if __name__=='__main__': unittest.main()
