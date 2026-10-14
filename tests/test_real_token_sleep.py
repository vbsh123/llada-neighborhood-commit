"""Small CPU fixtures matching LLaDA operations; no checkpoint or CUDA execution."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F

from experiments.token_sleep.selective import SelectiveForward
from experiments.token_sleep.simulation import FrozenStates
from experiments.token_sleep.benchmark import compact_rows,probability_comparison,validate,decode,evaluate,parser,validation_acceptance
from experiments.token_sleep.diagnostics import diagnose,difference


class Rotary(nn.Module):
    def get_rotary_embedding(self,n,device):
        freq = torch.arange(n,device=device).float()[:,None]*torch.tensor([1.,.01],device=device)[None,:]
        angles = torch.cat((freq,freq),-1)[None,None]
        return angles.sin(),angles.cos()
    def apply_rotary_pos_emb(self,sin,cos,x):
        a,b = x.chunk(2,-1)
        return (x*cos+torch.cat((-b,a),-1)*sin).to(x.dtype)
    def forward(self,q,k):
        sin,cos = self.get_rotary_embedding(k.shape[-2],q.device)
        return self.apply_rotary_pos_emb(sin,cos,q),self.apply_rotary_pos_emb(sin,cos,k)


class Block(nn.Module):
    def __init__(self,config):
        super().__init__(); self.config=config
        self.q_norm=self.k_norm=None
        self.rotary_emb=Rotary()
        self.attn_norm=nn.LayerNorm(8); self.ff_norm=nn.LayerNorm(8)
        self.q_proj=nn.Linear(8,8,bias=False)
        self.k_proj=nn.Linear(8,config.effective_n_kv_heads*4,bias=False)
        self.v_proj=nn.Linear(8,config.effective_n_kv_heads*4,bias=False)
        self.attn_out=nn.Linear(8,8,bias=False)
        self.ff_proj=nn.Linear(8,12,bias=False); self.up_proj=nn.Linear(8,12,bias=False)
        self.ff_out=nn.Linear(12,8,bias=False); self.act=nn.SiLU()
    def _scaled_dot_product_attention(self,q,k,v,attn_mask=None,dropout_p=0.,is_causal=False):
        if q.shape[1]!=k.shape[1]:
            repeat=q.shape[1]//k.shape[1]
            k=k.repeat_interleave(repeat,dim=1); v=v.repeat_interleave(repeat,dim=1)
        return F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal)
    def attention(self,q,k,v,attention_bias=None,layer_past=None,use_cache=False):
        b,n,_=q.shape
        q=q.view(b,n,2,4).transpose(1,2)
        k=k.view(b,n,self.config.effective_n_kv_heads,4).transpose(1,2)
        v=v.view(b,n,self.config.effective_n_kv_heads,4).transpose(1,2)
        q,k=self.rotary_emb(q,k)
        h=self._scaled_dot_product_attention(q,k,v).transpose(1,2).contiguous().view(b,n,8)
        return self.attn_out(h),None
    def forward(self,h,use_cache=False):
        norm=self.attn_norm(h)
        att,_=self.attention(self.q_proj(norm),self.k_proj(norm),self.v_proj(norm),use_cache=use_cache)
        h=h+att; norm=self.ff_norm(h)
        return h+self.ff_out(self.act(self.ff_proj(norm))*self.up_proj(norm)),None


class Model(nn.Module):
    device='cpu'
    def __init__(self,kv_heads=2):
        super().__init__()
        config=SimpleNamespace(alibi=False,rope=True,n_heads=2,effective_n_kv_heads=kv_heads,
            rope_full_precision=True,input_emb_norm=False,d_model=8,weight_tying=False,scale_logits=False,
            max_sequence_length=128)
        transformer=nn.ModuleDict({'wte':nn.Embedding(20,8),'emb_drop':nn.Identity(),
            'blocks':nn.ModuleList([Block(config),Block(config),Block(config)]),
            'ln_f':nn.LayerNorm(8),'ff_out':nn.Linear(8,20,bias=False)})
        self.model=nn.Module(); self.model.transformer=transformer
        self.config=config
    def forward(self,ids,use_cache=False):
        tr=self.model.transformer; h=tr.wte(ids)
        for block in tr.blocks: h,_=block(h,use_cache=use_cache)
        return SimpleNamespace(logits=tr.ff_out(tr.ln_f(h)))


class SelectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(57)
        self.ids=torch.tensor([[1,2,0,0,0,0,0]])

    def exercise(self,kv_heads):
        model=Model(kv_heads).eval(); blocks=model.model.transformer.blocks
        backend=SelectiveForward(model,blocks=blocks)
        with torch.inference_mode():
            raw=model(self.ids).logits
            first=backend(self.ids,[],[2,3,4,5,6])
            torch.testing.assert_close(first,raw[:,2:],rtol=1e-5,atol=1e-6)
            self.assertEqual(backend.last_active_count,7)
            # Simulator captures exactly the same initial cache state.
            with FrozenStates(blocks) as frozen:
                frozen.begin([]); model(self.ids); frozen.end()
                changed=self.ids.clone(); changed[0,3]=4
                cached=[(k.clone(),v.clone()) for k,v in backend.kv]
                counts=[]; heads=[]
                hook=blocks[0].q_proj.register_forward_pre_hook(lambda mod,args:counts.append(args[0].shape[1]))
                ffn=blocks[0].ff_proj.register_forward_pre_hook(lambda mod,args:counts.append(args[0].shape[1]))
                head=model.model.transformer.ff_out.register_forward_pre_hook(lambda mod,args:heads.append(args[0].shape[1]))
                try:
                    for _ in range(2):
                        frozen.begin([2,5]); sim=model(changed).logits; frozen.end()
                        with frozen.bypass(): real=backend(changed,[2,5],[4,6])
                        torch.testing.assert_close(real,sim[:,[4,6]],rtol=1e-5,atol=1e-6)
                        self.assertEqual(counts[-2:],[5,5])  # Real Q and FFN actually see 5 rows.
                        self.assertEqual(heads[-1],2)
                        for old,new in zip(cached,backend.kv):
                            for a,b in zip(old,new):
                                self.assertTrue(torch.equal(a[:,:,[2,5]],b[:,:,[2,5]]))
                    # Wake and freshly recompute all layers from embeddings.
                    with frozen.bypass():
                        wake=backend(changed,[],[2,4,5])
                        fresh=model(changed).logits
                    torch.testing.assert_close(wake,fresh[:,[2,4,5]],rtol=1e-5,atol=1e-6)
                finally:
                    hook.remove(); ffn.remove(); head.remove()
            # Original attention methods must be restored after every call.
            torch.testing.assert_close(model(changed).logits,fresh,rtol=0,atol=0)

    def test_full_and_sparse_original_position_attention(self): self.exercise(2)
    def test_grouped_query_attention_and_cache_layout(self): self.exercise(1)

    def test_initial_sleep_and_stale_prediction_rejected(self):
        model=Model().eval(); backend=SelectiveForward(model,blocks=model.model.transformer.blocks)
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'fully awake'): backend(self.ids,[2],[3])
            backend(self.ids,[],[2,3])
            with self.assertRaisesRegex(ValueError,'Cannot predict'): backend(self.ids,[2],[2])

    def test_prediction_mapping_and_mask_probability(self):
        logits=torch.zeros(1,2,20); logits[0,0,0]=5.; logits[0,0,3]=4.; logits[0,1,4]=6.
        rows=compact_rows(logits,[6,2],0)
        self.assertEqual([r['position'] for r in rows],[6,2])
        self.assertEqual([r['token_id'] for r in rows],[3,4])
        self.assertLess(rows[0]['confidence'],.5)
        comparison=probability_comparison(logits,logits,0)
        self.assertEqual(comparison['max_probability_difference'],0.)


class Tokenizer:
    eos_token_id=19
    def apply_chat_template(self,messages,**kwargs): return messages[0]['content']
    def encode(self,text,**kwargs): return [1,2]
    def get_vocab(self): return {}
    def decode(self,ids,**kwargs): return ' '.join(str(i) for i in ids if i not in (0,19))


class BenchmarkTests(unittest.TestCase):
    def test_scaled_gate_is_opt_in_and_does_not_ignore_selection_or_fp32_failures(self):
        report={'passed':False,'max_probability_tolerance':.02,
                'steps':[{'dense_selection_equal':True,'sleep_selection_equal':True,
                          'dense_vs_original':{'max_probability_difference':0.}}],
                'sparse_original_position_check':{'max_probability_difference':.016}}
        local={'layers':[{}],'all_fp32_scaled_checks_passed':True}
        self.assertFalse(validation_acceptance(report,local)['accepted'])
        self.assertTrue(validation_acceptance(report,local,True)['accepted'])
        self.assertFalse(validation_acceptance(report,None,True)['accepted'])
        local['all_fp32_scaled_checks_passed']=False
        self.assertFalse(validation_acceptance(report,local,True)['accepted'])
        local['all_fp32_scaled_checks_passed']=True
        report['steps'][0]['sleep_selection_equal']=False
        self.assertFalse(validation_acceptance(report,local,True)['accepted'])

    def test_scaled_error_uses_reference_size_and_rejects_nonfinite(self):
        big=torch.full((10,),1000.)
        small=torch.ones(10)
        self.assertTrue(difference(big+.001,big)['fp32_scaled_check_passed'])
        self.assertFalse(difference(small+.001,small)['fp32_scaled_check_passed'])
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            difference(torch.tensor([float('nan')]),torch.ones(1))

    def test_teacher_forced_validation_and_timed_decoder_plumbing(self):
        torch.manual_seed(81)
        model=Model().eval(); tokenizer=Tokenizer()
        config=dict(length=5,mask_id=0,sleep_confidence=.5,high_entropy=2.,high_sleep=2,
            validation_steps=4,validation_tolerance=.00001,policy='top1',commit_threshold=.9,
            warmup_forwards=1)
        sample=dict(id='00001',question='fixture',answer='#### 1')
        blocks=lambda m:list(m.model.transformer.blocks)
        with patch('experiments.token_sleep.selective.checked_blocks',blocks), \
             patch('experiments.token_sleep.benchmark.checked_blocks',blocks):
            report=validate(model,tokenizer,sample,config)
            self.assertTrue(report['passed'],report)
            self.assertIn('sparse_original_position_check',report)
            with tempfile.TemporaryDirectory() as temp, \
                 patch('torch.cuda.synchronize'),patch('torch.cuda.reset_peak_memory_stats'), \
                 patch('torch.cuda.max_memory_allocated',return_value=0):
                out=Path(temp); results={}
                for mode in ('original','active_baseline','real_sleep'):
                    directory=out/mode/'samples'/sample['id']; directory.mkdir(parents=True)
                    results[mode]=decode(model,tokenizer,sample,config,mode,directory)
                    self.assertEqual(results[mode]['steps'],5)
                    self.assertNotIn(0,results[mode]['final_ids'])
                    self.assertTrue((directory/'trace.jsonl.gz').exists())
                self.assertEqual(results['original']['final_ids'],results['active_baseline']['final_ids'])
                self.assertEqual(results['active_baseline']['vocabulary_head_rows'],15)
                self.assertEqual(results['original']['vocabulary_head_rows'],35)
                summary=evaluate(out,[sample],('original','active_baseline','real_sleep'))
                self.assertIn('active_baseline_vs_real_sleep',summary['comparisons'])
                self.assertAlmostEqual(summary['comparisons']['original_vs_active_baseline']['vocabulary_head_row_reduction_fraction'],1-15/35)
                default=evaluate(out,[sample])
                self.assertNotIn('original',default['modes'])
                self.assertEqual(set(default['comparisons']),{'active_baseline_vs_real_sleep'})

    def test_vanilla_benchmark_is_opt_in(self):
        self.assertFalse(parser().parse_args(['--out','fixture']).include_original)
        self.assertTrue(parser().parse_args(['--out','fixture','--include-original']).include_original)
        self.assertFalse(parser().parse_args(['--out','fixture']).allow_scaled_fp32_drift)

    def test_layer_diagnostic_identical_inputs_and_dtype_restoration(self):
        self.exercise_layer_diagnostic(torch.float32)

    def test_layer_diagnostic_bf16_weights_restored(self):
        self.exercise_layer_diagnostic(torch.bfloat16)

    def exercise_layer_diagnostic(self,dtype):
        torch.manual_seed(81)
        model=Model().eval().to(dtype=dtype); original={name:p.clone() for name,p in model.named_parameters()}
        config=dict(length=5,mask_id=0,sleep_confidence=.5,high_entropy=2.,high_sleep=2,
                    policy='top1',commit_threshold=.9)
        with patch('experiments.token_sleep.diagnostics.checked_blocks',lambda m:list(m.model.transformer.blocks)):
            report=diagnose(model,Tokenizer(),dict(question='fixture'),config)
        self.assertTrue(report['all_fp32_local_layers_within_tolerance'])
        self.assertTrue(report['all_fp32_scaled_checks_passed'])
        self.assertEqual(len(report['layers']),3)
        self.assertIn('q_proj',report['layers'][0]['fp32']['stages'])
        for name,p in model.named_parameters():
            self.assertTrue(torch.equal(p,original[name]))


if __name__=='__main__': unittest.main()
