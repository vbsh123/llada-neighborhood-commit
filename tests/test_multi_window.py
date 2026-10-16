"""CPU fixtures only: scheduling, real sparse math, shared validation/decoder."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from experiments.multi_window.schedule import WindowSchedule, merge_windows
from experiments.multi_window.run import parser, run
from experiments.token_sleep.benchmark import validate, decode, evaluate
from test_real_token_sleep import Model, Tokenizer


class ScheduleTests(unittest.TestCase):
    def test_merge_and_invalid_geometry(self):
        self.assertEqual(merge_windows([(8,10),(1,3),(4,6),(10,12)]),[(1,6),(8,12)])
        for kwargs in (dict(radius=-1),dict(refresh_every=0)):
            with self.assertRaises(ValueError): WindowSchedule(20,'simulated_sleep',**kwargs)

    def test_discovery_only_on_refresh_and_two_regions(self):
        s=WindowSchedule(30,'simulated_sleep',radius=1,refresh_every=4)
        state=[0]*30
        self.assertEqual(s.prepare(0,state,0),[])
        s.finish(0,[],[5]); state[5]=1
        self.assertEqual(s.windows,[(4,6)])
        sleepers=s.prepare(1,state,0)
        self.assertIn(20,sleepers); self.assertNotIn(4,sleepers)
        self.assertEqual(s.prepare(4,state,0),[])
        s.finish(4,[],[20]); state[20]=1
        self.assertEqual(s.windows,[(4,6),(19,21)])
        self.assertEqual(s.stats['windows_opened'],2)
        self.assertEqual(s.stats['global_refreshes'],2)
        self.assertNotIn(5,s.prepare(5,state,0))
        self.assertEqual(s.snapshot()['windows_before'],[[4,6],[19,21]])

    def test_expansion_merger_and_batch_births(self):
        s=WindowSchedule(20,'simulated_sleep',radius=2)
        s.prepare(0,[0]*20,0); s.finish(0,[],[3,10])
        self.assertEqual(s.windows,[(1,5),(8,12)])
        s.prepare(1,[0]*20,0); s.finish(1,[],[5,8])
        self.assertEqual(s.windows,[(1,12)])
        self.assertEqual(s.stats['windows_merged'],1)
        t=WindowSchedule(20,'simulated_sleep',radius=2)
        t.prepare(0,[0]*20,0); t.finish(0,[],[3,4])
        self.assertEqual(t.stats['windows_opened'],1)
        self.assertEqual(t.stats['new_region_seeds'],2)

    def test_retirement_fallback_and_filled_rows_sleep(self):
        s=WindowSchedule(10,'simulated_sleep',radius=1)
        s.prepare(0,[0]*10,0); s.finish(0,[],[2])
        # A filled position OUTSIDE the active window sleeps too.
        state=[0]*10; state[8]=1
        self.assertIn(8,s.prepare(1,state,0))
        self.assertEqual(s.stats['sleeping_filled_position_forwards'],1)
        state[1:4]=[1,1,1]
        self.assertEqual(s.prepare(2,state,0),[])
        self.assertTrue(s.fallback)
        self.assertEqual(s.stats['windows_retired'],1)
        self.assertEqual(s.snapshot()['refresh_reason'],'empty_windows')

    def test_baseline_refresh_one_and_edges(self):
        for mode,refresh in [('baseline',8),('simulated_sleep',1)]:
            s=WindowSchedule(8,mode,radius=2,refresh_every=refresh)
            for step in range(3):
                self.assertEqual(s.prepare(step,[0]*8,0),[])
                s.finish(step,[],[0,7])
            if mode!='baseline': self.assertEqual(s.windows,[(0,2),(5,7)])

    def test_last_commit_wakes_even_when_its_window_retires(self):
        s=WindowSchedule(12,'simulated_sleep',radius=1)
        state=[0]*12
        s.prepare(0,state,0); s.finish(0,[],[2,8])
        state[1:4]=[1,1,1]; state[8]=1
        # A final fill exhausted the first window while the other still has masks.
        s.prepare(1,[0]*12,0); s.finish(1,[],[2])
        sleepers=s.prepare(2,state,0)
        self.assertEqual(s.windows,[(7,9)])
        self.assertNotIn(2,sleepers)
        self.assertIn(1,sleepers)
        self.assertFalse(s.snapshot()['global_refresh'])

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
