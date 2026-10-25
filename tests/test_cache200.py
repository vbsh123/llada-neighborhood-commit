from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import test_cache
from audio_cache import prepare200
from audio_cache.data import fingerprint
from audio_cache.model import DIFFA_REVISION,LLM_REVISION
from audio_cache.run import parser,POLICIES


class PreparationTests(unittest.TestCase):
    def test_reservoir_matches_original_selection_algorithm(self):
        rows=list(range(1319))
        original=[]
        eligible=0
        rng=random.Random(20261024)
        for row in rows:
            eligible+=1
            slot=len(original) if len(original)<500 else rng.randrange(eligible)
            if slot<500:
                if slot==len(original):original.append(row)
                else:original[slot]=row
        self.assertEqual(prepare200.reservoir(rows,500,20261024),original)

    def test_fresh_cohort_excludes_all130_and_reconstructed500(self):
        payload=io.BytesIO()
        sf.write(payload,np.zeros(100,dtype=np.float32),16000,format='WAV',subtype='PCM_16')
        source='data/test-00000-of-00001.parquet'
        def rows(files):
            for index in range(1319):
                yield index,source,index,dict(answer=str(index),audio=dict(bytes=payload.getvalue()))
        with patch.object(prepare200,'source_rows',side_effect=rows),patch('huggingface_hub.HfApi') as api:
            api.return_value.list_repo_files.return_value=[source]
            reference=prepare200.fresh_reference(20261025)
        self.assertEqual(len(reference['rows']),200)
        self.assertEqual(len(reference['excluded_ids']),630)
        self.assertFalse({r['id'] for r in reference['rows']}&set(reference['excluded_ids']))
        self.assertEqual(reference['sha256'],fingerprint(reference['rows']))
        earlier=set(json.loads(Path(prepare200.__file__).with_name('earlier130_ids.json').read_text())['ids'])
        eligible=[i for i in range(1319) if f'gsm8k-speech-{i:05d}' not in earlier]
        old=prepare200.reservoir(eligible,500,20261024)
        self.assertEqual(set(reference['excluded_ids'])-earlier,{f'gsm8k-speech-{i:05d}' for i in old})

    def test_reused200_preserves_original_baselines_without_correctness_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[dict(id=f'gsm8k-speech-{i:05d}',prompt='question',target='18',metric='numeric',
                       source_dataset=prepare200.DATASET,source_revision=prepare200.REVISION,
                       source_file='data/test.parquet',source_row=i,audio_sha256='hash',audio='old.wav') for i in range(220)]
            config=dict(inputs=rows,settings=dict(seed=1729,length=512,block_length=32,steps=256,lora_alpha=16),
                resolved_revisions=dict(diffa=DIFFA_REVISION,llm=LLM_REVISION,whisper=prepare200.WHISPER_REVISION))
            (root/'config.json').write_text(json.dumps(config))
            for index,row in enumerate(rows):
                path=root/'samples'/f'{index:05d}'
                path.mkdir(parents=True)
                result=dict(id=row['id'],prompt=row['prompt'],target=row['target'],text='#### '+str(index),
                    answer_ids=[1,2,126081,31],benchmark=dict(baseline_visible_answer_tokens=2,
                        baseline_median_seconds=2.,baseline_forwards=4))
                (path/'result.json').write_text(json.dumps(result))
            reference,saved=prepare200.reused_reference(root,20261025)
            expected=random.Random(20261025).sample(range(220),200)
            self.assertEqual(reference['source_indices'],expected)
            self.assertEqual([r['id'] for r in reference['rows']],[rows[i]['id'] for i in expected])
            self.assertEqual(len(saved['measurements']),200)
            self.assertTrue(all(r['historical_timing'] for r in saved['measurements'].values()))
            self.assertNotIn('condition',next(iter(saved['measurements'].values())))
            self.assertEqual(prepare200.find_baseline_run([root],20261025),root.resolve())
            config['settings']['block_length']=64
            with self.assertRaisesRegex(ValueError,'decoder differs'):
                prepare200.verify_baseline_config(config)
            (root/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'No new baseline inference was started'):
                prepare200.find_baseline_run([root],20261025)


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'),'Requires pinned DIFFA checkout')
class Run200Tests(unittest.TestCase):
    setUp=test_cache.ModelTests.setUp

    def test_200_questions_all_eight_policies_and_one_baseline_each(self):
        from audio_cache import run
        rows=[dict(id=f'fresh{i}',audio='fixture.wav',prompt='question',target='18',metric='numeric',
            source_revision=prepare200.REVISION,audio_sha256='hash',cohort='cache_fresh200_v31') for i in range(200)]
        names=['323232','161616','888','16816','16168','323216','32328','321632']
        seen=[]
        def generated(llm,prepared,tokenizer,args,intervals=None,*,policy=None):
            seen.append(intervals)
            return torch.tensor([[1,2,3,30]]),dict(seconds=1.,visible_answer_tokens=3,visible_answer_ids=[4,5,6],
                answer_has_no_eos=False,peak_allocated_bytes=0,text='#### 18',accounting=[dict(forwards=8)])
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory)
            reference=root/'cohort.json'
            reference.write_text(json.dumps(dict(mode='fresh',rows=rows,sha256=fingerprint(rows))))
            stack.enter_context(patch.object(run,'read_manifest',return_value=rows))
            stack.enter_context(patch.object(run,'load_audio',return_value=(None,1.)))
            stack.enter_context(patch.object(run,'prepare',return_value=self.prepared))
            stack.enter_context(patch.object(run,'load_model',return_value=(SimpleNamespace(llm_model=self.llm),None,None,{})))
            stack.enter_context(patch.object(run,'generation',side_effect=generated))
            for name in ('is_available','is_bf16_supported'):
                stack.enter_context(patch('torch.cuda.'+name,return_value=True))
            stack.enter_context(patch('torch.cuda.manual_seed_all'))
            stack.enter_context(patch('torch.cuda.get_device_name',return_value='CPU fixture'))
            stack.enter_context(patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=100)))
            argv=['run','--manifest','fixture','--cohort-reference',str(reference),'--out',str(root/'run'),
                '--samples','200','--baseline-mode','fresh','--saved-baselines',str(root/'nonexistent.json'),
                '--diffa-root',os.environ['DIFFA_ROOT'],'--policies',*names]
            with patch.object(sys,'argv',argv):run.main()
            self.assertEqual(len(seen),9+200*9)
            self.assertEqual(seen.count(None),201)
            for name in names:self.assertEqual(seen.count(POLICIES[name]),201)
            summary=json.loads((root/'run'/'summary.json').read_text())
            self.assertEqual(len(summary),9)
            self.assertTrue(all(r['samples']==200 for r in summary.values()))
            self.assertTrue(all(r['timing_source']=='current' for r in summary.values()))
            self.assertTrue((root/'run'/'analysis'/'comparisons.csv').is_file())
            before=len(seen)
            with patch.object(sys,'argv',argv):run.main()
            self.assertEqual(len(seen),before)
            saved_path=root/'saved.json'
            baseline=dict(seconds=1.,visible_answer_tokens=3,visible_answer_ids=[4,5,6],answer_has_no_eos=False,
                          text='#### 18',accounting=[dict(forwards=8)],historical_timing=True)
            saved_path.write_text(json.dumps(dict(input_sha256=fingerprint(rows),measurements={r['id']:baseline for r in rows})))
            reused_argv=list(argv)
            reused_argv[reused_argv.index('--out')+1]=str(root/'reused')
            reused_argv[reused_argv.index('--baseline-mode')+1]='saved'
            reused_argv[reused_argv.index('--saved-baselines')+1]=str(saved_path)
            with patch.object(sys,'argv',reused_argv):run.main()
            self.assertEqual(len(seen)-before,8+200*8)
            self.assertEqual(seen.count(None),201)  # No additional baseline/warmup in saved mode.
            reused=json.loads((root/'reused'/'summary.json').read_text())
            self.assertTrue(all(r['samples']==200 for r in reused.values()))
            self.assertEqual(reused['baseline']['timing_source'],'historical')


if __name__=='__main__':unittest.main()
