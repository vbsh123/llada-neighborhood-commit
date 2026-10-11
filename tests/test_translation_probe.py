"""Translation input, decoder trace and resume checks with synthetic CPU logits."""
import contextlib
import gzip
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from confidence_geography.run import collect_sample, score_gsm8k
from experiments.translation_probe import run as runner
from experiments.translation_probe.evaluate import evaluate, score_translation


class Tokenizer:
    all_special_ids=[0,6,7]
    eos_token_id=6
    pieces=['[MASK]','The ','cat ','sleeps','x','y','[EOS]','[PROMPT]','z']
    def apply_chat_template(self,messages,**kwargs):return messages[0]['content']+'\n[ASSISTANT]'
    def encode(self,text,**kwargs):return [7]
    def get_vocab(self):return {}
    def convert_ids_to_tokens(self,i):return self.pieces[i]
    def decode(self,ids,skip_special_tokens=False,**kwargs):
        return ''.join(self.pieces[i] for i in ids if not skip_special_tokens or i not in self.all_special_ids)


class Model:
    device='cpu'
    config=SimpleNamespace(max_position_embeddings=32)
    def __call__(self,x,use_cache=False):
        assert not use_cache
        logits=torch.zeros(1,x.shape[1],9)
        for p,tid in enumerate((1,2,3,6)):logits[0,p+1,tid]=12-p
        return SimpleNamespace(logits=logits)


def collect_fixture(model,tokenizer,config,sample,directory,**kwargs):
    # Test logits use a nine-token vocabulary; no checkpoint or GPU is loaded.
    return collect_sample(model,tokenizer,{**config,'mask_id':0},sample,directory,**kwargs)


@unittest.skipUnless(importlib.util.find_spec('sacrebleu') is not None,
                     'Install the translation extra for translation tests')
class TranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def test_fixed_ten_pairs_and_exact_prompt(self):
        path=runner.HERE/'inputs/flores_zh_en_dev_10.jsonl'
        samples=runner.load_samples(path,10)
        provenance=json.loads((path.parent/'provenance.json').read_text())
        self.assertEqual(len(samples),10)
        self.assertEqual([s['flores_id'] for s in samples],provenance['flores_ids'])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),provenance['selected_inputs_sha256'])
        self.assertEqual(runner.load_samples(path,2),samples[:2])
        for sample in samples:
            self.assertEqual(sample['question'],runner.INSTRUCTION+sample['source_text'])
            self.assertNotIn(sample['answer'],sample['question'])
        with self.assertRaises(ValueError):runner.load_samples(path,11)

    def test_task_scores_do_not_use_numeric_accuracy(self):
        sample={'source_text':'猫在睡觉。','answer':'The cat sleeps'}
        score=score_translation('The cat sleeps',sample)
        self.assertEqual(score['chrfpp'],100)
        self.assertNotIn('correct_numeric',score)
        self.assertTrue(score_translation('',sample)['empty_translation'])
        self.assertTrue(score_gsm8k('The answer is 42',{'answer':'reasoning\n#### 42'})['correct_numeric'])

    def test_real_trace_writer_top1_threshold_and_fallback(self):
        sample={'id':'00001','question':runner.INSTRUCTION+'猫在睡觉。','source_text':'猫在睡觉。','answer':'The cat sleeps'}
        for policy,threshold,steps in [('top1',.9,4),('threshold',.9,1),('threshold',1.,4)]:
            with self.subTest(policy=policy,threshold=threshold), tempfile.TemporaryDirectory() as tmp:
                config={'mask_id':0,'length':4,'block_length':4,'top_k':5,'policy':policy,
                        'commit_threshold':threshold,'thresholds':[.9],'log_every':32}
                result=collect_sample(Model(),Tokenizer(),config,sample,Path(tmp),
                                      prompt_protocol=runner.PROMPT_PROTOCOL,score_output=score_translation)
                self.assertEqual(result['steps'],steps)
                self.assertEqual(result['translation'],'The cat sleeps')
                self.assertEqual(result['answer_token_length'],3)
                self.assertEqual(result['final_ids'],[1,2,3,6])
                self.assertEqual(result['chrfpp'],100)
                self.assertNotIn('correct_numeric',result)
                with gzip.open(Path(tmp)/'trace.jsonl.gz','rt') as handle:records=[json.loads(line) for line in handle]
                self.assertEqual(records[0]['prompt_protocol'],runner.PROMPT_PROTOCOL)
                self.assertEqual(records[0]['user_message'],sample['question'])
                self.assertEqual(records[-1],result)
                states=[r for r in records if r['type']=='step']
                self.assertEqual(len(states),steps)
                self.assertEqual(states[0]['state_ids'],[0,0,0,0])
                if steps==1:self.assertEqual(set(states[0]['commit_positions']),{0,1,2,3})

    def test_matched_runner_evaluation_resume_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs.jsonl'
            rows=[{'id':f'{i:05d}','source_language':'zho_Hans','target_language':'eng_Latn',
                   'source_text':'猫在睡觉。','reference':'The cat sleeps'} for i in (1,2)]
            inputs.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args=runner.parser().parse_args(['--out',str(root/'run'),'--inputs',str(inputs),'--samples','2','--length','4'])
            with contextlib.redirect_stdout(io.StringIO()), patch.object(runner,'load_model',return_value=(Model(),Tokenizer())) as load, \
                 patch.object(runner,'collect_sample',side_effect=collect_fixture), patch('torch.cuda.reset_peak_memory_stats'):
                runner.run(args)
                self.assertEqual(load.call_count,1)
            summary=json.loads((args.out/'comparison.json').read_text())
            self.assertEqual(summary['policies']['top1']['total_forwards'],8)
            self.assertEqual(summary['policies']['threshold']['total_forwards'],2)
            self.assertEqual(summary['policies']['top1']['chrfpp'],100)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(runner,'load_model',side_effect=AssertionError('Resume must not load model')):
                runner.run(args)
            with contextlib.redirect_stdout(io.StringIO()):report=evaluate(args.out)
            self.assertEqual(report,summary)
            args.commit_threshold=.8
            with self.assertRaisesRegex(ValueError,'arguments/source changed'):runner.run(args)

    def test_incorrect_result_reference_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'top1/samples/00001';directory.mkdir(parents=True)
            (root/'manifest.json').write_text(json.dumps({'config':{'policies':['top1']}}))
            (root/'samples.jsonl').write_text(json.dumps({'id':'00001','answer':'The cat sleeps','source_text':'猫在睡觉。'})+'\n')
            (directory/'result.json').write_text(json.dumps({'sample_id':'00001','reference':'Wrong','source_text':'猫在睡觉。'}))
            with self.assertRaisesRegex(ValueError,'Mismatched input/result'):evaluate(root)


if __name__=='__main__':unittest.main()
