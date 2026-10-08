import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from confidence_geography.matched_baseline import matched_inputs, run_matched_baseline
from confidence_geography.run import PROMPT_PROTOCOL, SCORING_PROTOCOL


class MatchedBaselineTests(unittest.TestCase):
    def fixture(self, root):
        samples = [{'id': '00009', 'dataset_index': 9, 'question': 'q9', 'answer': '#### 9'},
                   {'id': '00002', 'dataset_index': 2, 'question': 'q2', 'answer': '#### 2'}]
        config = {'policy': 'top1_window', 'window_size': 4, 'samples': 2, 'seed': 1729,
                  'prompt_protocol': PROMPT_PROTOCOL, 'scoring_protocol': SCORING_PROTOCOL,
                  'model': 'original-model', 'revision': 'original-model-commit'}
        root.mkdir()
        (root/'manifest.json').write_text(json.dumps({'config': config, 'dataset_info': {'revision': 'dataset-commit'}}))
        (root/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
        for sample in samples:
            folder=root/'samples'/sample['id'];folder.mkdir(parents=True)
            (folder/'trace.jsonl.gz').write_bytes(b'not opened by sample selection')
            (folder/'result.json').write_text(json.dumps({'sample_id': sample['id'],
                                                        'question': sample['question'], 'reference': sample['answer']}))
        return config, samples

    def test_exact_saved_order_ids_and_all_settings_except_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original,samples=self.fixture(root/'window')
            config,selected,provenance=matched_inputs(root/'window')
            self.assertEqual(selected,samples)
            self.assertEqual(config,{**original,'policy':'top1'})
            self.assertEqual(provenance['revision'],'dataset-commit')
            with patch('confidence_geography.matched_baseline.execute') as execute:
                run_matched_baseline(root/'window',root/'top1')
                self.assertEqual(execute.call_args.args[0]['policy'],'top1')
                self.assertEqual(execute.call_args.kwargs['matched_samples'],samples)

    def test_source_mismatch_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root/'window')
            result=root/'window/samples/00009/result.json'
            r=json.loads(result.read_text());r['question']='wrong';result.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                matched_inputs(root/'window')

    def test_resume_cannot_switch_question_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root/'window')
            out=root/'top1';out.mkdir()
            (out/'manifest.json').write_text('{}')
            (out/'samples.jsonl').write_text('{}\n')
            with patch('confidence_geography.matched_baseline.execute') as execute:
                with self.assertRaisesRegex(ValueError,'sample list differs'):
                    run_matched_baseline(root/'window',out)
                execute.assert_not_called()


if __name__=='__main__':unittest.main()
