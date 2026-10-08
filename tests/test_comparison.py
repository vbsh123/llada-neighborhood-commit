"""Paired result accounting using synthetic saved traces, with no inference."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from confidence_geography.compare_window import compare_window


class ComparisonTests(unittest.TestCase):
    def fixture(self,root,policy,question='q'):
        path=root/policy/'samples'/'00001';path.mkdir(parents=True)
        result={'sample_id':'00001','question':question,'reference':'#### 2',
                'correct_numeric':policy=='top1','steps':1,'elapsed_seconds':2,
                'hit_length_limit':False,'answer_token_length':2}
        (path/'result.json').write_text(json.dumps(result))
        positions=[{'position':0,'confidence':.99,'special':False},
                   {'position':1,'confidence':.4,'special':False},
                   {'position':2,'confidence':1.,'special':True}]
        records=[{'type':'header','config':{'policy':policy,'seed':1729}},
                 {'type':'step','forward_seconds':1.,'commit_positions':[0] if policy=='top1' else [0,1,2],
                  'positions':positions,'selection':{'anchor_position':0}}]
        with gzip.open(path/'trace.jsonl.gz','wt') as handle:
            for r in records:handle.write(json.dumps(r)+'\n')
        return root/policy

    def test_correctness_and_neighbor_population(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=self.fixture(root,'top1');b=self.fixture(root,'top1_window')
            result=compare_window(a,b,root/'comparison.json')
            self.assertEqual(result['paired_correctness'],{'baseline_only_correct':1})
            self.assertEqual(result['numeric_accuracy_change_percentage_points'],-100)
            self.assertEqual(result['top1_window']['forced_neighbor_count'],2)
            self.assertEqual(result['top1_window']['forced_answer_neighbors']['count'],1)
            self.assertEqual(result['top1_window']['forced_answer_neighbors']['below_0_9'],1)

    def test_mismatched_question_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=self.fixture(root,'top1');b=self.fixture(root,'top1_window','different')
            with self.assertRaisesRegex(ValueError,'Question/reference mismatch'):
                compare_window(a,b,root/'comparison.json')

    def test_repair_comparison_includes_extra_compute_and_checks_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=self.fixture(root,'top1');b=self.fixture(root,'top1_window')
            repaired=root/'repaired';folder=repaired/'samples/00001';folder.mkdir(parents=True)
            original=json.loads((b/'samples/00001/result.json').read_text())
            (folder/'original_result.json').write_text(json.dumps(original))
            (repaired/'manifest.json').write_text(json.dumps({'source_config':{'policy':'top1_window','seed':1729},'repair_threshold':.75}))
            result={**original,'correct_numeric':True,'repair_forwards':1,'repair_forward_seconds':.25,'elapsed_seconds':2.5}
            (folder/'result.json').write_text(json.dumps(result))
            report=compare_window(a,b,root/'comparison.json',repaired)
            r=report['repaired_window']
            self.assertEqual(r['total_forwards_including_generation'],2)
            self.assertEqual(r['total_forward_seconds_including_generation'],1.25)
            self.assertEqual(r['numeric_accuracy'],1.)
            self.assertEqual(r['paired_vs_top1'],{'both_correct':1})
            original['correct_numeric']=True
            (folder/'original_result.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'different original'):
                compare_window(a,b,root/'comparison.json',repaired)


if __name__=='__main__':unittest.main()
