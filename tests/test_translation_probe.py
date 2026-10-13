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
from experiments.translation_probe.render import collect, visible_text


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

    def test_invisible_output_detection(self):
        for text in ('', ' \n', '\u200b', '\u200b\u200c'):
            self.assertFalse(visible_text(text))
        self.assertTrue(visible_text('Translation\u200b'))
        self.assertTrue(visible_text('猫'))

    def test_four_sentence_inputs_preserve_alignment(self):
        directory=runner.HERE/'inputs_four_sentences'
        rows=[json.loads(line) for line in (directory/'flores_zh_en_dev_10.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),10)
        ids=[]
        for row in rows:
            pairs=row['sentence_pairs']
            self.assertEqual(len(pairs),4)
            self.assertEqual(row['source_text'],'\n'.join(p['source_text'] for p in pairs))
            self.assertEqual(row['reference'],'\n'.join(p['reference'] for p in pairs))
            self.assertEqual(row['flores_ids'],[p['flores_id'] for p in pairs])
            ids.extend(row['flores_ids'])
        self.assertEqual(len(set(ids)),40)
        provenance=json.loads((directory/'provenance.json').read_text())
        self.assertEqual(provenance['flores_ids'],ids)
        self.assertEqual(provenance['selected_inputs_sha256'],hashlib.sha256((directory/'flores_zh_en_dev_10.jsonl').read_bytes()).hexdigest())

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

    def test_explicit_four_sentence_instruction(self):
        path=runner.HERE/'inputs_four_sentences/flores_zh_en_dev_10.jsonl'
        default=runner.load_samples(path,2)
        diagnostic=runner.load_samples(path,2,runner.ALL_INSTRUCTION)
        for a,b in zip(default,diagnostic):
            self.assertEqual(b['question'],runner.ALL_INSTRUCTION+a['source_text'])
            self.assertEqual(a['answer'],b['answer'])
            self.assertEqual(a['source_text'],b['source_text'])
            self.assertEqual(a['id'],b['id'])
            self.assertNotIn(b['answer'],b['question'])
        self.assertFalse(runner.parser().parse_args(['--out','unused']).translate_all)

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
            reports,audit=collect(args.out)
            self.assertEqual(len(reports),4)
            self.assertEqual(audit['identical_translation_pairs'],2)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(runner,'load_model',side_effect=AssertionError('Resume must not load model')):
                runner.run(args)
            with contextlib.redirect_stdout(io.StringIO()):report=evaluate(args.out)
            self.assertEqual(report,summary)
            args.commit_threshold=.8
            with self.assertRaisesRegex(ValueError,'arguments/source changed'):runner.run(args)

    def test_blockwise_translation_defers_remote_eos(self):
        class EarlyEOSModel(Model):
            def __call__(self,x,use_cache=False):
                logits=torch.zeros(1,x.shape[1],9)
                for p,tid in enumerate((1,2,3,6)):logits[0,p+1,tid]=16-p
                logits[0,4,6]=24
                if x[0,4].item()==6 and x[0,1].item()==0:
                    logits[0,1,8]=25
                return SimpleNamespace(logits=logits)
        sample={'id':'00001','question':'Translate all four sentences.','source_text':'猫在睡觉。','answer':'The cat sleeps'}
        for policy in ('top1','threshold'):
            with self.subTest(policy=policy),tempfile.TemporaryDirectory() as tmp:
                config={'mask_id':0,'length':4,'block_length':2,'top_k':5,'policy':policy,
                        'commit_threshold':.9,'thresholds':[.9],'log_every':32}
                result=collect_sample(EarlyEOSModel(),Tokenizer(),config,sample,Path(tmp),
                                      prompt_protocol=runner.ALL_PROMPT_PROTOCOL,score_output=score_translation)
                self.assertEqual(result['translation'],'The cat sleeps')
                with gzip.open(Path(tmp)/'trace.jsonl.gz','rt') as f:steps=[json.loads(l) for l in f if '"type":"step"' in l]
                self.assertTrue(steps)
                self.assertTrue(all(p<2 for p in steps[0]['commit_positions']))
                self.assertTrue(all(s['block_start']<=p<s['block_end'] for s in steps for p in s['commit_positions']))
        with tempfile.TemporaryDirectory() as tmp:
            global_config={**config,'block_length':4,'policy':'top1'}
            result=collect_sample(EarlyEOSModel(),Tokenizer(),global_config,sample,Path(tmp),
                                  prompt_protocol=runner.ALL_PROMPT_PROTOCOL,score_output=score_translation)
            self.assertEqual(result['translation'],'zcat sleeps')
        with tempfile.TemporaryDirectory() as tmp:
            args=runner.parser().parse_args(['--out',tmp,'--length','4','--block-length','3'])
            with self.assertRaisesRegex(ValueError,'block-length'):runner.run(args)

    def test_runner_requires_full_window_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=runner.parser().parse_args(['--out',tmp,'--samples','2','--length','4','--block-length','2'])
            with patch.object(runner,'load_model',side_effect=AssertionError('Must reject before loading')):
                with self.assertRaisesRegex(ValueError,'full-window'):runner.run(args)
            self.assertFalse((Path(tmp)/'manifest.json').exists())
            args.block_length=4
            with contextlib.redirect_stdout(io.StringIO()),patch.object(runner,'load_model',return_value=(Model(),Tokenizer())),patch.object(runner,'collect_sample',side_effect=collect_fixture),patch('torch.cuda.reset_peak_memory_stats'):
                runner.run(args)
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(manifest['config']['block_length'],4)
            self.assertEqual(manifest['config']['length'],4)
            self.assertEqual(manifest['config']['stop_policy'],'unchanged')
            reports,audit=collect(Path(tmp))
            self.assertEqual(len(reports),4)
            self.assertEqual(audit['threshold']['all_forwards'],2)

    def test_stop_deferral_keeps_remote_positions_available_and_raw_confidence(self):
        rows=[{'position':p,'token_id':tid,'confidence':prob,'eligible':True}
              for p,tid,prob in [(0,1,.6),(2,6,.99),(200,2,.95),(255,6,.999)]]
        for policy in ('top1','threshold'):
            config={'policy':policy,'commit_threshold':.9,'stop_policy':'unchanged'}
            chosen,metadata=runner.select_translation(rows,config,{6})
            self.assertEqual(chosen[0]['position'],255)
            self.assertEqual(metadata['stop_selection']['deferred_positions'],[])
            config['stop_policy']='defer'
            chosen,metadata=runner.select_translation(rows,config,{6})
            self.assertEqual([r['position'] for r in chosen],[200])
            self.assertEqual(chosen[0]['confidence'],.95)
            self.assertEqual(set(metadata['stop_selection']['deferred_positions']),{2,255})
            config['commit_threshold']=1.
            chosen,_=runner.select_translation(rows,config,{6})
            self.assertEqual(chosen[0]['position'],200)  # Ordinary fallback, any distance.
            only_stop=[r for r in rows if r['token_id']==6]
            chosen,metadata=runner.select_translation(only_stop,config,{6})
            self.assertEqual(chosen[0]['position'],255)
            self.assertEqual(metadata['stop_selection']['deferred_positions'],[])

    def test_stop_deferral_prevents_early_eos_context_change(self):
        class EarlyEOSModel(Model):
            def __call__(self,x,use_cache=False):
                logits=torch.zeros(1,x.shape[1],9)
                for p,tid in enumerate((1,2,3,6)):logits[0,p+1,tid]=16-p
                logits[0,4,6]=24
                if x[0,4].item()==6 and x[0,1].item()==0:logits[0,1,8]=25
                return SimpleNamespace(logits=logits)
        sample={'id':'00001','question':'Translate.','source_text':'猫在睡觉。','answer':'The cat sleeps'}
        for policy in ('top1','threshold'):
            for stop_policy in ('unchanged','defer'):
                with self.subTest(policy=policy,stop_policy=stop_policy),tempfile.TemporaryDirectory() as tmp:
                    config={'mask_id':0,'length':4,'block_length':4,'top_k':5,'policy':policy,
                            'commit_threshold':.9,'thresholds':[.9],'log_every':32,'stop_policy':stop_policy}
                    result=collect_sample(EarlyEOSModel(),Tokenizer(),config,sample,Path(tmp),
                        prompt_protocol=runner.PROMPT_PROTOCOL,score_output=score_translation,select_output=runner.select_translation)
                    if stop_policy=='defer':self.assertEqual(result['translation'],'The cat sleeps')
                    with gzip.open(Path(tmp)/'trace.jsonl.gz','rt') as f:steps=[r for r in map(json.loads,f) if r['type']=='step']
                    self.assertTrue(all((r['block_start'],r['block_end'])==(0,4) for r in steps))
                    if stop_policy=='defer':
                        self.assertNotIn(3,steps[0]['commit_positions'])
                        self.assertEqual(steps[0]['stop_selection']['deferred_positions'],[3])

    def test_diagnostic_runner_manifest_resume_and_viewer_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=runner.parser().parse_args(['--out',tmp,'--samples','2','--length','4','--stop-policy','defer'])
            with contextlib.redirect_stdout(io.StringIO()),patch.object(runner,'load_model',return_value=(Model(),Tokenizer())),patch.object(runner,'collect_sample',side_effect=collect_fixture),patch('torch.cuda.reset_peak_memory_stats'):
                runner.run(args)
            reports,audit=collect(Path(tmp))
            self.assertEqual(len(reports),4)
            self.assertTrue(audit['top1']['all_states_and_selections_verified'])
            self.assertTrue(audit['threshold']['all_states_and_selections_verified'])
            with contextlib.redirect_stdout(io.StringIO()),patch.object(runner,'load_model',side_effect=AssertionError('Resume must not load model')):
                runner.run(args)
            args.stop_policy='unchanged'
            with self.assertRaisesRegex(ValueError,'arguments/source changed'):runner.run(args)

    def test_incorrect_result_reference_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'top1/samples/00001';directory.mkdir(parents=True)
            (root/'manifest.json').write_text(json.dumps({'config':{'policies':['top1']}}))
            (root/'samples.jsonl').write_text(json.dumps({'id':'00001','answer':'The cat sleeps','source_text':'猫在睡觉。'})+'\n')
            (directory/'result.json').write_text(json.dumps({'sample_id':'00001','reference':'Wrong','source_text':'猫在睡觉。'}))
            with self.assertRaisesRegex(ValueError,'Mismatched input/result'):evaluate(root)


if __name__=='__main__':unittest.main()
