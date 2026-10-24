"""Independent dense references and full overnight orchestration checks."""
from contextlib import ExitStack
from copy import deepcopy
import json
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

import test_cache
from audio_cache.cache import PrefixCache
from audio_cache.data import COHORT
from audio_cache.decode import forward_logits
from audio_cache.pruning import GlobalPruning
from audio_cache.sparse import AudioSparseAttention


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'),'Requires pinned DIFFA checkout')
class InferenceTests(unittest.TestCase):
    setUp=test_cache.ModelTests.setUp

    @torch.no_grad()
    def test_global_query_scores_against_full_softmax(self):
        captured=[]
        with GlobalPruning(self.llm,self.prepared,threshold=1e-20,interval=1,starts=(1,1)) as engine:
            original=engine.observe
            def record(q,k,layer):
                mean=(q.float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1).mean((0,1,2))
                captured.append(mean)
                original(q,k,layer)
            engine.observe=record
            forward_logits(self.llm,self.prepared,torch.tensor([[1,2,3,4,31,31,31]]))
        check=engine.accounting()['pruning']['checks'][0]
        self.assertEqual(check['query_count_per_layer'],self.prepared.answer_start+4)
        expected=torch.stack(captured).mean(0)
        for group in ('acoustic','prompt','semantic'):
            self.assertAlmostEqual(check['attention_mass_by_group'][group],
                                   float(expected[self.prepared.group_positions(group)].sum()),places=6)
        self.assertFalse(engine.pending)

    @torch.no_grad()
    def test_compaction_matches_independent_dense_key_removal_original_rope(self):
        for dtype in (torch.float32,torch.bfloat16):
            tested=deepcopy(self.llm).to(dtype)
            reference=deepcopy(self.llm).to(dtype)
            omitted=[self.prepared.group_positions('acoustic')[0],self.prepared.group_positions('semantic')[-1]]
            clock=[0]
            for block in reference.model.transformer.blocks:
                original=block._scaled_dot_product_attention
                def attend(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_original=original):
                    if clock[0]:
                        keep=torch.tensor([i for i in range(k.shape[-2]) if i not in omitted])
                        k,v=k.index_select(-2,keep),v.index_select(-2,keep)
                    return _original(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False)
                block._scaled_dot_product_attention=attend
            with GlobalPruning(tested,self.prepared,threshold=1e-20,interval=1,starts=(1,1)) as engine:
                for step in range(4):
                    clock[0]=step
                    x=torch.tensor([[1,2,3,4+step,31,31,31]])
                    actual=forward_logits(tested,self.prepared,x)
                    expected=forward_logits(reference,self.prepared,x)
                    tolerance=.02 if dtype==torch.bfloat16 else 2e-6
                    torch.testing.assert_close(actual,expected,atol=tolerance,rtol=tolerance)
                    if step==0:
                        engine.pending=[dict(step=1,score_step=0,group='acoustic',drop_budget=2,
                            dropped_positions=omitted,dropped_by_group=dict(acoustic=1,semantic=1,prompt=0))]
            self.assertGreater(engine.accounting()['skipped_token_update_fraction'],0)
            self.assertEqual(engine.position_ids,tuple(p for p in range(engine.length) if p not in omitted))

    @torch.no_grad()
    def test_generic_matches_audio_applied_event_times_and_counts(self):
        x=torch.tensor([[1,2,3,31,31,31,31]])
        with GlobalPruning(self.llm,self.prepared,threshold=100,interval=1,starts=(1,1),patience=1,fraction=.5) as audio:
            for step in range(5):
                forward_logits(self.llm,self.prepared,x)
        source=audio.accounting()['pruning']
        with GlobalPruning(self.llm,self.prepared,control='generic',source=source,interval=1,starts=(1,1)) as generic:
            for step in range(5):
                forward_logits(self.llm,self.prepared,x)
        compare=lambda engine:[(r['step'],r['group'],r['drop_budget']) for r in engine.accounting()['pruning']['selections']]
        self.assertEqual(compare(audio),compare(generic))
        self.assertTrue(source['selections'])
        self.assertTrue(all(p<self.prepared.answer_start for r in generic.selections for p in r['dropped_positions']))

    def dense_sparse_mask(self,sparse,group,queries):
        semantic=self.prepared.group_positions('semantic')
        anchors=set(sparse.anchors[group])
        acoustic=set(self.prepared.group_positions('acoustic'))
        allowed=torch.zeros(len(queries),sparse.length,dtype=torch.bool)
        for row,p in enumerate(queries):
            for key in range(sparse.length):
                if p in acoustic or p in anchors or key not in semantic or key in anchors:
                    allowed[row,key]=True
                elif p in semantic:
                    allowed[row,key]=abs(semantic.index(p)//sparse.chunk_size-semantic.index(key)//sparse.chunk_size)<=sparse.neighbors
        return allowed

    @torch.no_grad()
    def test_sparse_gathered_calls_match_independent_dense_masks_and_anchor_scores(self):
        with AudioSparseAttention(self.llm,self.prepared,4,chunk_size=2,dense_layers=0,layer_group=1,
                                  anchor_fractions=(.5,.25)) as sparse:
            for step in range(3):
                for layer,block in enumerate(self.llm.model.transformer.blocks):
                    q,k,v=[torch.randn(1,2,sparse.length,8) for _ in range(3)]
                    actual=block._scaled_dot_product_attention(q,k,v)
                    if step==0:
                        expected=F.scaled_dot_product_attention(q,k,v)
                        semantic=torch.tensor(self.prepared.group_positions('semantic'))
                        logits=q.float()@k[:,:,semantic].float().transpose(-1,-2)/math.sqrt(8)
                        logits.masked_fill_(torch.arange(sparse.length)[:,None]==semantic[None,:],-torch.inf)
                        scores=logits.softmax(-1).sum((0,1,2))
                        anchors=[]
                        for start in range(0,len(semantic),2):
                            end=min(start+2,len(semantic))
                            count=math.ceil((end-start)*sparse.fractions[layer])
                            pick=torch.argsort(scores[start:end],descending=True,stable=True)[:count]+start
                            anchors+=semantic[pick].tolist()
                        self.assertEqual(sparse.anchors[layer],tuple(sorted(anchors)))
                    else:
                        mask=self.dense_sparse_mask(sparse,layer,range(sparse.length))
                        expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            self.assertGreater(sparse.accounting()['skipped_attention_pair_fraction'],0)

    @torch.no_grad()
    def test_full_pinned_sparse_cached_reference_and_group_coordinates(self):
        for dtype in (torch.float32,torch.bfloat16):
            tested=deepcopy(self.llm).to(dtype)
            reference=deepcopy(self.llm).to(dtype)
            with ExitStack() as stack:
                cached=stack.enter_context(PrefixCache(tested,self.prepared,(2,2,2)))
                sparse=stack.enter_context(AudioSparseAttention(tested,self.prepared,4,cache=cached,
                    chunk_size=2,dense_layers=0,layer_group=1,anchor_fractions=(.5,.25)))
                refcache=stack.enter_context(PrefixCache(reference,self.prepared,(2,2,2)))
                for layer,block in enumerate(reference.model.transformer.blocks):
                    original=block._scaled_dot_product_attention
                    def attend(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_layer=layer):
                        mask=None if refcache.step==0 else self.dense_sparse_mask(sparse,_layer,refcache.active_ids_by_layer[_layer])
                        return F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                    block._scaled_dot_product_attention=attend
                    stack.callback(setattr,block,'_scaled_dot_product_attention',original)
                for step in range(5):
                    x=torch.tensor([[1,2,3,4+step,31,31,31]])
                    actual=forward_logits(tested,self.prepared,x)
                    expected=forward_logits(reference,self.prepared,x)
                    tolerance=.02 if dtype==torch.bfloat16 else 2e-6
                    torch.testing.assert_close(actual,expected,atol=tolerance,rtol=tolerance)
                self.assertEqual(sparse.accounting()['forwards'],5)

    @torch.no_grad()
    def test_sparse_actual32_layer_groups_first_dense_then_cached(self):
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
        config=deepcopy(self.llm.config)
        config.n_layers=32
        tested=LLaDAModelLM(config,init_params=True).eval()
        reference=deepcopy(tested)
        clock=[0]
        with ExitStack() as stack:
            cache=stack.enter_context(PrefixCache(tested,self.prepared,(32,32,32)))
            sparse=stack.enter_context(AudioSparseAttention(tested,self.prepared,4,cache=cache))
            refcache=stack.enter_context(PrefixCache(reference,self.prepared,(32,32,32)))
            for layer,block in enumerate(reference.model.transformer.blocks):
                original=block._scaled_dot_product_attention
                def attend(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_layer=layer):
                    mask=None
                    if clock[0] and _layer>=8:
                        mask=self.dense_sparse_mask(sparse,(_layer-8)//8,refcache.active_ids_by_layer[_layer])
                    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                block._scaled_dot_product_attention=attend
                stack.callback(setattr,block,'_scaled_dot_product_attention',original)
            for step in range(2):
                clock[0]=step
                x=torch.tensor([[1,2,3,4+step,31,31,31]])
                torch.testing.assert_close(forward_logits(tested,self.prepared,x),
                    forward_logits(reference,self.prepared,x),atol=3e-6,rtol=3e-6)
            self.assertEqual(set(sparse.anchors),{0,1,2})
            n=len(self.prepared.group_positions('semantic'))
            self.assertEqual([len(sparse.anchors[i]) for i in range(3)],[math.ceil(n*f) for f in (.5,.25,.125)])

    def test_all_three_suite_clis_saved_baseline_and_combined_report(self):
        from audio_cache import run
        from audio_cache.night_report import summarize
        rows=[{**r,'audio':'fixture.wav'} for r in json.loads(COHORT.read_text())['rows']]
        seen=[]
        def generated(llm,prepared,tokenizer,args,intervals=None,*,policy=None):
            seen.append((args.suite,intervals,policy))
            account=dict(forwards=8,pruning=dict(checks=[],selections=[]))
            return torch.tensor([[1,2,3,30]]),dict(seconds=1.,visible_answer_tokens=3,visible_answer_ids=[4,5,6],
                answer_has_no_eos=False,peak_allocated_bytes=0,text='#### 18',accounting=[account])
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory)
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
            for suite in ('cache','prune','sparse'):
                argv=['run','--suite',suite,'--manifest','fixture','--out',str(root/suite),'--samples','1',
                      '--diffa-root',os.environ['DIFFA_ROOT']]
                if suite=='cache':argv+=['--policies','323232','321632','323216']
                with patch.object(sys,'argv',argv):
                    run.main()
                self.assertEqual(len(seen),2*sum(3 if s=='cache' else 2 for s in ('cache','prune','sparse')[:('cache','prune','sparse').index(suite)+1]))
                before=len(seen)
                with patch.object(sys,'argv',argv):
                    run.main()
                self.assertEqual(len(seen),before)
            self.assertEqual(len(seen),14)  # One warmup and one measured decode per condition; NO baseline.
            summarize(root)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(len(summary),8)  # Seven current policies plus saved reference.
            self.assertIn('sparse_audio_cache323232',summary)
            self.assertIn('prune_global_generic',summary)
            self.assertTrue(all(intervals is not None or policy is not None for _,intervals,policy in seen))
            generic=[p for _,_,p in seen if p and p.get('pruning',{}).get('control')=='generic']
            self.assertTrue(all('source' in p['pruning'] for p in generic))


if __name__=='__main__':
    unittest.main()
