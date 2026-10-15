"""Offline trace semantics: bucket boundaries, elapsed forwards and bootstrap scope."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.token_sleep.analyze_readiness import (
    answer_positions, buckets, collect_baseline, histogram_quantile, summarize,
    nearby_reveal_during_sleep)


class ReadinessTests(unittest.TestCase):
    def test_bucket_endpoints(self):
        self.assertEqual(buckets(0, 0), (0, 1))
        self.assertEqual(buckets(.1, 1), (1, 2))
        self.assertEqual(buckets(.5, 2), (5, 3))
        self.assertEqual(buckets(1, 3), (9, 4))
        with self.assertRaises(ValueError):
            buckets(.3, float('nan'))

    def test_histogram_median_and_empty(self):
        self.assertEqual(histogram_quantile(np.array([1, 0, 1]), .5), 1.)
        self.assertIsNone(histogram_quantile(np.zeros(3), .5))

    def test_nearby_reveal_affects_next_forward_not_same_forward(self):
        episode=dict(position=5,start=0,actual_wake=2)
        commits={p:10 for p in range(12)}
        commits[6]=0
        self.assertTrue(nearby_reveal_during_sleep(episode,commits,12))
        commits[6]=1
        self.assertFalse(nearby_reveal_during_sleep(episode,commits,12))
        episode['actual_wake']=3
        self.assertTrue(nearby_reveal_during_sleep(episode,commits,12))
        commits[6]=10;commits[10]=0
        self.assertFalse(nearby_reveal_during_sleep(episode,commits,12))

    def test_sleep_decision_excludes_same_forward_commitment(self):
        hist=np.zeros((2,10,5,5),dtype=np.int64)
        hist[0,2,0,[0,1,4]]=[1,2,1]
        hist[1,2,0,[0,2]]=[2,1]
        sums=np.zeros((2,2,10,5,2))
        result=summarize(hist,sums,replicates=100)
        all_cell=next(c for c in result['all'] if c['confidence_bucket']==2 and c['entropy_band']==0)
        wait_cell=next(c for c in result['waiting'] if c['confidence_bucket']==2 and c['entropy_band']==0)
        self.assertEqual(all_cell['observations'],7)
        self.assertEqual(wait_cell['observations'],4)
        self.assertEqual(wait_cell['median'],1.5)
        self.assertEqual(wait_cell['within']['1'],50.)
        self.assertEqual(wait_cell['within']['2'],75.)
        self.assertEqual(wait_cell['within_ci95']['256'],[100.,100.])
        self.assertEqual(wait_cell['questions'],2)

    def test_final_answer_filter_and_real_step_difference(self):
        result=dict(sample_id='00001',final_ids=[5,6,7,8],first_stop_position=3,
                    token_dictionary={'5':{'special':False},'6':{'special':False},
                                      '7':{'special':True},'8':{'special':True}},
                    question='fixture',reference='#### 1',answer='1',correct_numeric=True)
        self.assertEqual(answer_positions(result),{0,1})
        def row(p,c,h):
            return dict(position=p,confidence=c,entropy=h,eligible=True)
        records=[dict(type='header',prompt_protocol='question_only_chat_v1',
                      config=dict(policy='top1',length=4,block_length=4,model='fixture',revision='pinned')),
                 dict(type='step',step=0,commit_positions=[3],commit_ids=[8],
                      positions=[row(0,.2,3),row(1,.6,1),row(2,.4,2),row(3,.9,0)]),
                 dict(type='step',step=1,commit_positions=[1],commit_ids=[6],
                      positions=[row(0,.3,2),row(1,.9,0),row(2,.4,2)]),
                 dict(type='step',step=2,commit_positions=[0],commit_ids=[5],
                      positions=[row(0,.9,0),row(2,.4,2)]),
                 dict(type='step',step=3,commit_positions=[2],commit_ids=[7],positions=[row(2,.9,0)])]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'top1'/'samples'/'00001';path.mkdir(parents=True)
            (path/'result.json').write_text(json.dumps(result))
            with gzip.open(path/'trace.jsonl.gz','wt') as f:
                for r in records:
                    f.write(json.dumps(r)+'\n')
            hist,sums,questions=collect_baseline(Path(temp),'top1')
        self.assertEqual(hist[:, :, 0].sum(),5)
        self.assertEqual(hist[0,2,4,2],1)
        self.assertEqual(hist[0,3,3,1],1)
        self.assertEqual(hist[0,6,2,1],1)
        self.assertEqual(hist[0,9,1,0],2)
        self.assertEqual(questions['00001']['valid_positions'],2)
        self.assertAlmostEqual(sums[0,1,2,4,0],.2)


if __name__=='__main__':
    unittest.main()
