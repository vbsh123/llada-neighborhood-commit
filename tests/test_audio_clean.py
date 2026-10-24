"""Independent dense references for global scoring and gathered audio attention."""
from contextlib import ExitStack
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments.audio_drift.decode import Prepared, forward_logits
from experiments.audio_drift.compact import PrefixCache
from experiments.audio_drift.progressive_pruning import progressive_conditions
from experiments.audio_clean.pruning import GlobalPruning
from experiments.audio_clean.sparse import AudioSparseAttention
from experiments.audio_clean.run import parser, specifications, ordered, summarize, write_json, generation


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'), 'Requires pinned DIFFA checkout')
class CleanTests(unittest.TestCase):
    def setUp(self):
        from experiments.audio_drift.checkout import verify_checkout
        source = verify_checkout(os.environ['DIFFA_ROOT'])
        sys.dont_write_bytecode = True
        sys.path.insert(0,str(source))
        from src.fast_dllm.model.configuration_llada import LLaDAConfig
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
        torch.set_num_threads(2)
        torch.manual_seed(81)
        config = LLaDAConfig(d_model=16,n_heads=2,n_kv_heads=2,n_layers=2,
            mlp_hidden_size=32,vocab_size=32,embedding_size=32,block_type='llama',block_group_size=1,
            rope=True,activation_type='silu',max_sequence_length=600,flash_attention=False,
            weight_tying=False,mask_token_id=31,eos_token_id=30,attention_dropout=0.,
            residual_dropout=0.,embedding_dropout=0.)
        self.llm = LLaDAModelLM(config,init_params=True).eval()
        self.prepared = Prepared(torch.tensor([[1,2,3]]),torch.randn(1,21,16),1,1,31)

    def dense_pattern(self, sparse, group, queries):
        n = sparse.length
        semantic = set(self.prepared.group_positions('semantic'))
        acoustic = set(self.prepared.group_positions('acoustic'))
        anchors = set(sparse.anchors[group])
        allowed = torch.zeros(len(queries),n,dtype=torch.bool)
        for row,p in enumerate(queries):
            for key in range(n):
                if p in acoustic or p in anchors or key not in semantic or key in anchors:
                    allowed[row,key] = True
                elif p in semantic:
                    own = self.prepared.group_positions('semantic').index(p)//sparse.chunk_size
                    other = self.prepared.group_positions('semantic').index(key)//sparse.chunk_size
                    allowed[row,key] = abs(own-other)<=sparse.neighbors
        return allowed

    @torch.no_grad()
    def test_gathered_attention_matches_independent_dense_mask_and_anchor_search(self):
        sparse = AudioSparseAttention(self.llm,self.prepared,4,dense_layers=0,layer_group=1,
            anchor_fractions=(.5,.25),chunk_size=3,proxy_queries=128)
        blocks = list(self.llm.model.transformer.blocks)
        originals = [b._scaled_dot_product_attention for b in blocks]
        n = sparse.length
        with sparse:
            for step in range(3):
                for layer,block in enumerate(blocks):
                    q,k,v = [torch.randn(1,2,n,8) for _ in range(3)]
                    actual = block._scaled_dot_product_attention(q,k,v)
                    if step == 0:
                        expected = F.scaled_dot_product_attention(q,k,v)
                        # Independent semantic-only proxy, diagonal removed.
                        sem = torch.tensor(self.prepared.group_positions('semantic'))
                        logits = q.float() @ k[:,:,sem].float().transpose(-1,-2)/math.sqrt(8)
                        logits.masked_fill_(torch.arange(n)[:,None]==sem[None,:],-torch.inf)
                        scores = logits.softmax(-1).sum((0,1,2))
                        chosen = []
                        for start in range(0,len(sem),3):
                            end = min(start+3,len(sem))
                            count = math.ceil((end-start)*sparse.fractions[layer])
                            order = torch.argsort(scores[start:end],descending=True,stable=True)[:count]+start
                            chosen += sem[order].tolist()
                        self.assertEqual(sparse.anchors[layer],tuple(sorted(chosen)))
                    else:
                        allowed = self.dense_pattern(sparse,layer,range(n))
                        expected = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed)
                    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            self.assertGreater(sparse.accounting()['skipped_attention_pair_fraction'],0)
        for block,original in zip(blocks,originals):
            self.assertEqual(block._scaled_dot_product_attention,original)

    @torch.no_grad()
    def test_full_pinned_model_cache_sparse_dense_reference(self):
        reference = deepcopy(self.llm)
        schedule = [dict(group='prefix',refresh_every=2)]
        tested_cache = PrefixCache(self.llm,self.prepared,schedule)
        reference_cache = PrefixCache(reference,self.prepared,schedule)
        sparse = AudioSparseAttention(self.llm,self.prepared,4,cache=tested_cache,
            dense_layers=0,layer_group=1,anchor_fractions=(.5,.25),chunk_size=3)
        seen = []
        with ExitStack() as stack:
            stack.enter_context(tested_cache)
            stack.enter_context(sparse)
            stack.enter_context(reference_cache)
            for layer,block in enumerate(reference.model.transformer.blocks):
                original = block._scaled_dot_product_attention
                def dense(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_layer=layer):
                    if reference_cache.step == 0:
                        mask = None
                    else:
                        positions = reference_cache.active_ids_by_layer[_layer]
                        mask = self.dense_pattern(sparse,_layer,positions)
                        seen.append((reference_cache.step,len(positions)))
                    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                block._scaled_dot_product_attention = dense
                stack.callback(setattr,block,'_scaled_dot_product_attention',original)
            for step in range(5):
                x = torch.tensor([[1,2,3,31,step+4,31,31]])
                actual = forward_logits(self.llm,self.prepared,x)
                expected = forward_logits(reference,self.prepared,x)
                tolerance = 2e-2 if actual.dtype == torch.bfloat16 else 2e-6
                torch.testing.assert_close(actual,expected,atol=tolerance,rtol=tolerance)
            self.assertIn((1,4),seen)  # Only answer queries on prefix-cache reuse.
            self.assertIn((2,sparse.length),seen)  # Whole prefix on refresh.
            self.assertEqual(sparse.accounting()['forwards'],5)

    def test_full_pinned_bfloat16_cache_sparse_reference(self):
        self.llm = self.llm.to(torch.bfloat16)
        self.test_full_pinned_model_cache_sparse_dense_reference()

    @torch.no_grad()
    def test_all_query_scores_match_full_attention_and_masked_scope(self):
        for scope,expected_count in [('all',self.prepared.answer_start+4),('all_masked',3),('current_block',1)]:
            spec = progressive_conditions(length=4,block_length=2,steps=4,acoustic_start=1,
                semantic_start=1,interval=1,patience=1,threshold=1e-20,fraction=.5)[0]['pruning']
            spec.update(query_scope=scope,query_chunk=2)
            engine = GlobalPruning(self.llm,self.prepared,spec)
            captured = []
            with engine:
                # Wrap the monitoring path only to record independently computed
                # full softmax scores; production still performs chunked scoring.
                scorer = engine.attention_scores
                def observe(q,k):
                    ids = engine.query_local
                    self.assertEqual(len(ids),expected_count)
                    expected = (q[:,:,ids].float() @ k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1).mean((0,1,2))
                    actual = scorer(q,k)
                    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
                    captured.append(expected)
                    return actual
                engine.attention_scores = observe
                forward_logits(self.llm,self.prepared,torch.tensor([[1,2,3,4,31,31,31]]))
            self.assertEqual(len(captured),2)
            check = engine.accounting()['pruning']['trigger_checks'][0]
            expected = torch.stack(captured).mean(0)[:self.prepared.answer_start]
            torch.testing.assert_close(torch.tensor(check['attention_mean']),expected)

    @torch.no_grad()
    def test_global_scoring_after_physical_removal_and_deep_layers(self):
        for depth in (0,1):
            spec = progressive_conditions(length=4,block_length=2,steps=4,acoustic_start=1,
                semantic_start=1,interval=1,patience=1,threshold=100,fraction=.5,layer_start=depth)[0]['pruning']
            spec.update(query_scope='all',query_chunk=3)
            with GlobalPruning(self.llm,self.prepared,spec) as engine:
                for step in range(3):
                    forward_logits(self.llm,self.prepared,torch.tensor([[1,2,3,31,31,31,31]]))
            account = engine.accounting()
            self.assertGreater(account['skipped_token_update_fraction'],0)
            self.assertTrue(account['pruning']['selections'])
            for check in account['pruning']['trigger_checks']:
                self.assertEqual(len(check['positions']),len(check['attention_mean']))

    def test_suites_exact_controls_dependencies_and_paired_reports(self):
        args = parser().parse_args(['--suite','cache','--manifest','pool','--out','out'])
        self.assertEqual([s['condition'] for s in specifications(args)],
            ['cache_a8_p8_s8','cache_a16_p8_s16','cache_a16_p16_s16'])
        args.suite = 'prune'
        args.compare_current_block = True
        specs = specifications(args)
        self.assertEqual(len(specs),4)
        for sample in ('a','b','c'):
            done = set()
            for spec in ordered(specs,sample,1729):
                self.assertTrue(not spec.get('control_source') or spec['control_source'] in done)
                done.add(spec['condition'])
        args.suite = 'sparse'
        self.assertEqual(len(specifications(args)),3)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'samples'/'00000').mkdir(parents=True)
            def row(name,correct,seconds,tokens):
                return dict(condition=name,secondary_correct=correct,score=dict(correct=correct),
                    secondary_parse_failed=False,seconds=seconds,visible_answer_tokens=tokens,
                    accounting=[dict(forwards=8)])
            write_json(out/'samples'/'00000'/'result.json',dict(id='a',baseline=row('baseline',True,2,10),
                policies=[row('cache',False,1,8)]))
            summarize(out)
            import json
            summary = json.loads((out/'summary.json').read_text())['cache']
            self.assertEqual(summary['accuracy_delta_pp'],-100)
            self.assertAlmostEqual(summary['tps_change_percent'],60)
            self.assertEqual(summary['decode_speedup'],2)
            self.assertEqual(summary['losses'],1)

    @torch.no_grad()
    def test_single_pass_generation_sparse_and_global_record_formats(self):
        # Run actual clean runner generation, including EOS and minimal records,
        # on the pinned tiny transformer. Mock only CUDA timing calls.
        args = SimpleNamespace(progress_every=0)
        tokenizer = SimpleNamespace(eos_token_id=30,decode=lambda ids,**kwargs:'#### 1')
        # Force EOS predictions for a quick decode with the real 512-slot canvas.
        output = torch.nn.Linear(16,32,bias=True)
        output.weight.zero_()
        output.bias.zero_()
        output.bias[30] = 100
        self.llm.model.transformer.ff_out = output
        spec = progressive_conditions()[0]
        spec['pruning'].update(query_scope='all',query_chunk=8)
        with patch('torch.cuda.synchronize'),patch('torch.cuda.reset_peak_memory_stats'),patch('torch.cuda.max_memory_allocated',return_value=0):
            final, measurement = generation(self.llm,self.prepared,tokenizer,args,spec)
        self.assertEqual(measurement['visible_answer_tokens'],0)
        self.assertFalse(measurement['answer_has_no_eos'])
        self.assertEqual(measurement['accounting'][0]['pruning']['query_scope'],'all')

    def test_clean_cli_all_suites_reports_and_resume(self):
        from experiments.audio_clean import run
        import json
        rows = [dict(id=f'q{i}',audio='audio.wav',prompt='solve',target='1',metric='numeric',
                     task='gsm8k_speech') for i in range(50)]
        def generated(llm,prepared,tokenizer,args,spec=None):
            account = dict(forwards=8,pruning=dict(selections=[],trigger_checks=[]))
            return torch.tensor([[1,2,3,30]]),dict(seconds=1.,visible_answer_tokens=3,
                visible_answer_ids=[4,5,6],answer_has_no_eos=False,peak_allocated_bytes=0,
                text='#### 1',accounting=[account])
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(run,'read_manifest',return_value=rows))
            stack.enter_context(patch.object(run,'fingerprint',return_value=run.DEV50_SHA))
            stack.enter_context(patch.object(run,'load_audio',return_value=(None,1.)))
            stack.enter_context(patch.object(run,'prepare',return_value=self.prepared))
            stack.enter_context(patch.object(run,'load_model',return_value=(SimpleNamespace(llm_model=self.llm),None,None,{})))
            mocked_generation = stack.enter_context(patch.object(run,'generation',side_effect=generated))
            for name in ('is_available','is_bf16_supported'):
                stack.enter_context(patch('torch.cuda.'+name,return_value=True))
            stack.enter_context(patch('torch.cuda.manual_seed_all'))
            stack.enter_context(patch('torch.cuda.get_device_name',return_value='CPU fixture'))
            stack.enter_context(patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=100)))
            for suite in ('cache','prune','sparse'):
                out = Path(directory)/suite
                argv = ['clean','--suite',suite,'--manifest','pool.jsonl','--out',str(out),
                        '--samples','1','--verify-samples','0','--diffa-root',os.environ['DIFFA_ROOT']]
                with patch.object(sys,'argv',argv):
                    run.main()
                result = json.loads((out/'samples'/'00000'/'result.json').read_text())
                self.assertEqual(len(result['policies']),2 if suite=='prune' else 3)
                self.assertTrue((out/'analysis'/'pairs.csv').is_file())
                self.assertTrue((out/'summary.json').is_file())
                calls = mocked_generation.call_count
                with patch.object(sys,'argv',argv):
                    run.main()
                self.assertEqual(mocked_generation.call_count,calls)


if __name__ == '__main__':
    unittest.main()
