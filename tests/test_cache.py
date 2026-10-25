import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from audio_cache.cache import PrefixCache
from audio_cache.data import COHORT, fingerprint, prepare_manifest, read_manifest
from audio_cache.decode import Prepared, decode, forward_logits
from audio_cache.extractor import extract_answer
from audio_cache.run import parser, generation


class DataTests(unittest.TestCase):
    def test_exact_prior50_and_default_policy(self):
        cohort=json.loads(COHORT.read_text())
        self.assertEqual(len(cohort['rows']),50)
        self.assertEqual(fingerprint(cohort['rows']),'8ed78386500a79daecdf9b9be573efe2c2a4001a5c667b23c58555a6e03205f2')
        self.assertEqual(cohort['sha256'],fingerprint(cohort['rows']))
        self.assertEqual(parser().parse_args([]).policies,['161616'])

    def test_restore_original_rows_and_validate_every_audio_hash(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from audio_cache import data
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            wave=np.linspace(-.2,.2,100,dtype=np.float32)
            encoded=io.BytesIO()
            sf.write(encoded,wave,16000,format='WAV',subtype='PCM_16')
            payload=encoded.getvalue()
            parquet=root/'test.parquet'
            pq.write_table(pa.Table.from_pylist([dict(audio=dict(bytes=payload)) for _ in range(50)]),parquet)
            rows=[dict(id=str(i),source_revision='pinned',prompt='question',target='1',metric='numeric',
                audio_sha256=hashlib.sha256(payload).hexdigest(),source_dataset='test',
                source_file='data/test.parquet',source_row=i) for i in range(50)]
            reference=root/'cohort.json'
            reference.write_text(json.dumps(dict(rows=rows,sha256=fingerprint(rows))))
            out=root/'manifest.jsonl'
            with patch.object(data,'COHORT',reference),patch('huggingface_hub.hf_hub_download',return_value=str(parquet)):
                prepare_manifest(out)
                restored=read_manifest(out)
                self.assertEqual([r['id'] for r in restored],[str(i) for i in range(50)])
                prepare_manifest(out)  # Resume validates without changing selection.
                Path(restored[3]['audio']).write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError,'Audio hash mismatch'):
                    read_manifest(out)

    def test_frozen_numeric_extraction(self):
        for text,answer in [('#### 120','120'),('So, the total cost is **$2400**.','2400'),
                            ('#### unknown',None),('First calculate 2+3=5.',None)]:
            self.assertEqual(extract_answer(text)[0],answer)

    def test_zero_token_report(self):
        from audio_cache.report import summarize
        from audio_cache.storage import write_json
        measurement=dict(condition='baseline',secondary_correct=False,score=dict(correct=False),
            secondary_parse_failed=True,seconds=1.,visible_answer_tokens=0,accounting=[dict(forwards=1)])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'samples'/'00000').mkdir(parents=True)
            write_json(out/'samples'/'00000'/'result.json',dict(id='empty',baseline=measurement,policies=[]))
            self.assertIn('N/A',summarize(out))


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'),'Requires pinned DIFFA checkout')
class ModelTests(unittest.TestCase):
    def setUp(self):
        from audio_cache.checkout import verify_checkout
        root=verify_checkout(os.environ['DIFFA_ROOT'])
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(root))
        from src.fast_dllm.model.configuration_llada import LLaDAConfig
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
        torch.set_num_threads(2)
        torch.manual_seed(23)
        config=LLaDAConfig(d_model=16,n_heads=2,n_kv_heads=2,n_layers=2,mlp_hidden_size=32,
            vocab_size=32,embedding_size=32,block_type='llama',block_group_size=1,rope=True,
            activation_type='silu',max_sequence_length=600,flash_attention=False,weight_tying=False,
            mask_token_id=31,eos_token_id=30,attention_dropout=0.,residual_dropout=0.,embedding_dropout=0.)
        self.llm=LLaDAModelLM(config,init_params=True).eval()
        self.prepared=Prepared(torch.tensor([[1,2,3]]),torch.randn(1,7,16),1,2,31)

    @torch.no_grad()
    def check_dense_reference(self,dtype):
        from copy import deepcopy
        for intervals in ((1,1,1),(8,8,8),(16,8,16),(16,16,16),(16,16,8),
                          (32,32,32),(32,16,32),(32,32,16),(32,32,8)):
            tested=deepcopy(self.llm).to(dtype)
            reference=deepcopy(self.llm).to(dtype)
            banks={}
            clock=[0]
            groups=[self.prepared.group_positions(g) for g in ('acoustic','prompt','semantic')]
            for layer,block in enumerate(reference.model.transformer.blocks):
                attention=block._scaled_dot_product_attention
                def capture(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_layer=layer,_attention=attention):
                    # Independent reference: compute every row, substitute stale
                    # KV columns, then restore the inactive block output rows.
                    inactive=[p for group,interval in zip(groups,intervals) if clock[0]%interval for p in group]
                    if inactive:
                        k,v=k.clone(),v.clone()
                        k[:,:,inactive]=banks[_layer][0][:,:,inactive]
                        v[:,:,inactive]=banks[_layer][1][:,:,inactive]
                    hidden=banks[_layer][2] if _layer in banks else None
                    banks[_layer]=[k.clone(),v.clone(),hidden]
                    return _attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False)
                block._scaled_dot_product_attention=capture
                def restore(module,args,output,_layer=layer):
                    inactive=[p for group,interval in zip(groups,intervals) if clock[0]%interval for p in group]
                    hidden=output[0].clone()
                    if inactive:
                        hidden[:,inactive]=banks[_layer][2][:,inactive]
                    banks[_layer][2]=hidden.clone()
                    return hidden,output[1]
                block.register_forward_hook(restore)
            with PrefixCache(tested,self.prepared,intervals) as cache:
                for step in range(34):
                    clock[0]=step
                    x=torch.tensor([[1,2,3,4+step%20,31,31,31]])
                    actual=forward_logits(tested,self.prepared,x)
                    expected=forward_logits(reference,self.prepared,x)
                    tolerance=.02 if dtype==torch.bfloat16 else 2e-6
                    torch.testing.assert_close(actual,expected,atol=tolerance,rtol=tolerance)
            self.assertEqual(cache.accounting()['forwards'],34)
            self.assertEqual(cache.accounting()['refresh_counts']['prompt'],len(range(0,34,intervals[1])))

    def test_cache_against_independent_dense_reference_float32(self):
        self.check_dense_reference(torch.float32)

    def test_cache_against_independent_dense_reference_bfloat16(self):
        self.check_dense_reference(torch.bfloat16)

    @torch.no_grad()
    def test_decoder_matches_pinned_official_and_identity_cache(self):
        from src.origin_generate import generate
        output=torch.nn.Linear(16,32,bias=True)
        output.bias[31]=-100
        self.llm.model.transformer.ff_out=output
        self.llm=self.llm.to(torch.bfloat16)
        baseline,_=decode(self.llm,self.prepared,4,2,4)
        official=generate(SimpleNamespace(llm_model=self.llm,device=self.llm.device),self.prepared.prompt,
            self.prepared.prompt.new_tensor([self.prepared.insertion]),self.prepared.speech,
            steps=4,gen_length=4,block_length=2,temperature=0.,cfg_scale=0.,mask_id=31)
        self.assertTrue(torch.equal(baseline,official))
        with PrefixCache(self.llm,self.prepared,(1,1,1)):
            identity,_=decode(self.llm,self.prepared,4,2,4)
        self.assertTrue(torch.equal(identity,baseline))

    @torch.no_grad()
    def test_runner_generation_actual_pinned_model(self):
        output=torch.nn.Linear(16,32,bias=True)
        output.weight.zero_()
        output.bias.zero_()
        output.bias[30]=100
        self.llm.model.transformer.ff_out=output
        args=SimpleNamespace(progress_every=0)
        tokenizer=SimpleNamespace(eos_token_id=30,decode=lambda ids,**kwargs:'#### 1')
        with patch('torch.cuda.synchronize'),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.max_memory_allocated',return_value=0):
            final,measurement=generation(self.llm,self.prepared,tokenizer,args,(16,16,16))
        self.assertEqual(measurement['visible_answer_tokens'],0)
        self.assertFalse(measurement['answer_has_no_eos'])
        self.assertGreater(measurement['accounting'][0]['forwards'],0)

    def test_default_cli_only161616_and_resume(self):
        from contextlib import ExitStack
        from audio_cache import run
        rows=[{**r,'audio':'fixture.wav'} for r in json.loads(COHORT.read_text())['rows']]
        seen=[]
        def generated(llm,prepared,tokenizer,args,intervals=None):
            seen.append(intervals)
            return torch.tensor([[1,2,3,30]]),dict(seconds=1.,visible_answer_tokens=3,
                visible_answer_ids=[4,5,6],answer_has_no_eos=False,peak_allocated_bytes=0,
                text='#### 18',accounting=[dict(forwards=8)])
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
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
            out=Path(directory)/'run'
            argv=['cache','--manifest','pool.jsonl','--out',str(out),'--samples','1',
                  '--verify-samples','0','--diffa-root',os.environ['DIFFA_ROOT']]
            with patch.object(sys,'argv',argv):
                run.main()
            self.assertEqual(seen,[(16,16,16),(16,16,16)])
            result=json.loads((out/'samples'/'00000'/'result.json').read_text())
            self.assertEqual([p['condition'] for p in result['policies']],['cache_a16_p16_s16'])
            self.assertEqual(result['id'],rows[0]['id'])
            with patch.object(sys,'argv',argv):
                run.main()
            self.assertEqual(len(seen),2)
            self.assertTrue(result['baseline']['historical_timing'])
            summary=json.loads((out/'summary.json').read_text())
            self.assertIsNone(summary['cache_a16_p16_s16']['decode_speedup'])
            reused=Path(directory)/'reused'
            reuse_argv=[*argv]
            reuse_argv[reuse_argv.index('--out')+1]=str(reused)
            reuse_argv+=['--reuse-run',str(out)]
            with patch.object(sys,'argv',reuse_argv):
                run.main()
            self.assertEqual(len(seen),2)
            reused_result=json.loads((reused/'samples'/'00000'/'result.json').read_text())
            self.assertEqual(reused_result['policies'],result['policies'])
            self.assertIn('reused_from',reused_result)


if __name__=='__main__':
    unittest.main()
