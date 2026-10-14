"""Scoring saved text only: no checkpoint, torch, datasets or GPU required."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.gsm8k_eval.scoring import HERE,COMMIT,load_protocol,score_saved
from experiments.gsm8k_eval.__main__ import evaluate


class ProtocolTests(unittest.TestCase):
    def test_pinned_components_remain_verbatim(self):
        provenance,_,_=load_protocol()
        self.assertEqual(provenance['commit'],COMMIT)
        source=(HERE/'snapshot.py').read_text()
        nodes={node.name:node for node in ast.parse(source).body if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
        for component in provenance['selected_components']:
            snippet=ast.get_source_segment(source,nodes[component['name']])
            self.assertEqual(hashlib.sha256(snippet.encode()).hexdigest(),component['source_sha256'])

    def test_flexible_and_strict_follow_harness_rules(self):
        answers=['The answer is 42.','#### 42','#### 42.0','#### 12\nCorrection: #### 99',
                 '#### 12\nI used 7 steps.','No answer','The answer is $1,234.','#### -5']
        references=['#### 42','#### 42','#### 42','#### 12','#### 12','#### 1','#### 1234','#### -5']
        scores=score_saved(answers,references)
        self.assertEqual(scores['strict-match']['correct'],[False,True,False,True,True,False,False,True])
        self.assertEqual(scores['flexible-extract']['correct'],[True,True,False,False,False,False,True,True])
        self.assertEqual(scores['strict-match']['predictions'][3],'12')
        self.assertEqual(scores['flexible-extract']['predictions'][3],'99')
        self.assertEqual(scores['flexible-extract']['predictions'][5],'[invalid]')
        self.assertEqual(scores['flexible-extract']['accuracy'],.5)

    def test_bad_reference_and_empty_input_rejected(self):
        with self.assertRaisesRegex(ValueError,'Reference lacks'):
            score_saved(['42'],['unmarked reference 42'])
        with self.assertRaises(ValueError): score_saved([],[])

    def test_no_torch_import(self):
        code="from experiments.gsm8k_eval.scoring import score_saved; import sys; score_saved(['42'], ['#### 42']); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)


class SavedResultTests(unittest.TestCase):
    def fixture(self,root):
        samples=[dict(id='00001',question='First?',answer='#### 42'),
                 dict(id='00002',question='Second?',answer='#### 12')]
        (root/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
        for mode,answers in [('active_baseline',['42','#### 12']),('real_sleep',['#### 42','#### 99'])]:
            for sample,answer in zip(samples,answers):
                directory=root/mode/'samples'/sample['id'];directory.mkdir(parents=True)
                result=dict(sample_id=sample['id'],question=sample['question'],reference=sample['answer'],
                            answer=answer,correct_numeric=(answer!='#### 99'))
                (directory/'result.json').write_text(json.dumps(result))

    def test_re_scores_existing_outputs_without_modifying_them(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            before={p:p.read_bytes() for p in root.rglob('result.json')}
            result=evaluate(root)
            self.assertEqual(result['samples_scored_per_mode'],2)
            self.assertEqual(result['modes']['active_baseline']['metrics']['exact_match,flexible-extract']['accuracy'],1.)
            self.assertEqual(result['modes']['active_baseline']['metrics']['exact_match,strict-match']['accuracy'],.5)
            self.assertEqual(result['paired_comparisons']['active_baseline_vs_real_sleep']['exact_match,flexible-extract']['paired_correctness']['baseline_only_correct'],1)
            self.assertTrue((root/'gsm8k_harness_eval'/'per_sample.csv').exists())
            for path,raw in before.items(): self.assertEqual(path.read_bytes(),raw)

    def test_incomplete_requires_explicit_common_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            (root/'real_sleep'/'samples'/'00002'/'result.json').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):evaluate(root)
            result=evaluate(root,allow_incomplete=True)
            self.assertEqual(result['sample_ids'],['00001'])
            self.assertEqual(result['missing_result_ids_by_mode']['real_sleep'],['00002'])
            self.assertEqual(result['excluded_completed_ids_by_mode']['active_baseline'],['00002'])

    def test_reference_mismatch_is_not_silently_compared(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            path=root/'real_sleep'/'samples'/'00001'/'result.json'
            result=json.loads(path.read_text());result['reference']='#### 99';path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'reference differs'):evaluate(root)

    def test_single_mode_can_be_scored(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            result=evaluate(root/'real_sleep')
            self.assertEqual(list(result['modes']),['real_sleep'])
            self.assertEqual(result['paired_comparisons'],{})


if __name__=='__main__': unittest.main()
