"""CPU operator/parity audit; never download checkpoints or run GPU generation."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from experiments.token_sleep.selective import SelectiveForward
from experiments.multi_window.profile import reference_backend, read_trace, aggregate, run, parser
from test_real_token_sleep import Model, Rotary, Tokenizer


class SharedRotary(Rotary):
    def __init__(self):
        super().__init__()
        self.tables={}
    def get_rotary_embedding(self,n,device):
        key=(n,str(device))
        if key not in self.tables:
            self.tables[key]=super().get_rotary_embedding(n,device)
        return self.tables[key]


class AuditTests(unittest.TestCase):
    def model_and_backends(self,dtype=torch.float32):
        torch.manual_seed(91)
        model=Model().eval().to(dtype=dtype)
        rotary=SharedRotary()
        for block in model.model.transformer.blocks: block.rotary_emb=rotary
        legacy,_=reference_backend()
        return model,legacy(model,blocks=model.model.transformer.blocks),SelectiveForward(
            model,blocks=model.model.transformer.blocks)

    def test_old_new_bitwise_parity_over_wakes_and_sparse_patterns(self):
        for dtype in (torch.float32,torch.bfloat16):
            with self.subTest(dtype=dtype):
                model,old,new=self.model_and_backends(dtype)
                x=torch.tensor([[1,2,0,0,0,0,0,0,0,0,0,0]])
                with torch.inference_mode():
                    for index,(sleepers,predictions) in enumerate([
                            ([],[2,3,7,10]),([2,7,9],[11,5,3]),
                            ([3,8],[10,4,2]),([],[11,3,5])]):
                        x[0,4]=index+3
                        a,b=old(x,sleepers,predictions),new(x,sleepers,predictions)
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                        for (ka,va),(kb,vb) in zip(old.kv,new.kv):
                            torch.testing.assert_close(ka,kb,rtol=0,atol=0)
                            torch.testing.assert_close(va,vb,rtol=0,atol=0)

    def test_redundant_index_operations_removed(self):
        model,old,new=self.model_and_backends()
        x=torch.tensor([[1,2,0,0,0,0,0,0,0,0,0,0]])
        def counts(backend,sleepers):
            class CountOps(TorchDispatchMode):
                def __init__(self): self.counts=Counter()
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    self.counts[func._schema.name]+=1
                    return func(*args,**(kwargs or {}))
            with CountOps() as p:
                backend(x,sleepers,[3,5,11])
            return p.counts
        with torch.inference_mode():
            for backend in (old,new): backend(x,[],[3,5,11])
            old_dense,new_dense=counts(old,[]),counts(new,[])
            old_sparse,new_sparse=counts(old,[2,7,9]),counts(new,[2,7,9])
        self.assertEqual(old_dense.get('aten::searchsorted',0),1)
        self.assertEqual(new_dense.get('aten::searchsorted',0),0)
        self.assertEqual(new_sparse.get('aten::searchsorted',0),0)
        # Embedding lookup may itself dispatch index_select. Compare the exact
        # reduction: six full-RoPE gathers + identity input gather, and four
        # redundant sparse-RoPE gathers across three shared-buffer layers.
        self.assertEqual(old_dense['aten::index_select']-new_dense['aten::index_select'],7)
        self.assertEqual(old_sparse['aten::index_select']-new_sparse['aten::index_select'],4)

    def test_different_rotary_buffers_not_reused_across_layers(self):
        model,old,new=self.model_and_backends()
        for block in model.model.transformer.blocks: block.rotary_emb=SharedRotary()
        x=torch.tensor([[1,2,0,0,0,0,0,0]])
        with torch.inference_mode():
            old(x,[],[2,5]); new(x,[],[2,5])
            torch.testing.assert_close(old(x,[3,7],[5,2]),new(x,[3,7],[5,2]),rtol=0,atol=0)

    def test_replay_reconstructs_states_and_rejects_sleeping_filled_rows(self):
        records=[{'type':'header','config':{'mask_id':0,'length':4},'prompt_ids':[1,2]},
                 {'type':'step','step':0,'sleep_positions':[],
                  'commit_positions':[1],'commit_ids':[3]},
                 {'type':'step','step':1,'sleep_positions':[0],
                  'commit_positions':[2],'commit_ids':[4]}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.jsonl.gz'
            def write():
                with gzip.open(path,'wt') as h:
                    for r in records: h.write(json.dumps(r)+'\n')
            write(); header,frames=read_trace(path,2)
            self.assertEqual(frames[1]['state'],[0,3,0,0])
            self.assertEqual(frames[1]['awake'],[2,3])
            records[2]['sleep_positions']=[1]; write()
            with self.assertRaisesRegex(ValueError,'only MASKs sleep'): read_trace(path,2)

    def test_timing_summary_partitions_geometry(self):
        rows=[dict(mode=mode,kind=kind,wall_seconds=.01,cuda_ms=9.)
              for mode in ('original','legacy_selective','updated_selective')
              for kind in ('all_awake','sparse')]
        result=aggregate(rows)
        self.assertEqual(result['updated_selective']['all']['calls'],2)
        self.assertEqual(result['updated_selective']['sparse']['mean_wall_ms'],10.)

    def test_full_replay_plumbing_on_cpu_with_fake_timers(self):
        class Event:
            def __init__(self,**kwargs): pass
            def record(self): pass
            def synchronize(self): pass
            def elapsed_time(self,other): return 1.
        config=dict(mask_id=0,length=4,policy='top1',commit_threshold=.9)
        records=[{'type':'header','config':config,'prompt_ids':[1,2]},
                 {'type':'step','step':0,'sleep_positions':[],
                  'commit_positions':[1],'commit_ids':[3]},
                 {'type':'step','step':1,'sleep_positions':[0],
                  'commit_positions':[2],'commit_ids':[4]}]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'saved'; source.mkdir()
            (source/'manifest.json').write_text(json.dumps({'config':config}))
            (source/'samples.jsonl').write_text(json.dumps({'id':'fixture'})+'\n')
            folder=source/'multi_window'/'samples'/'fixture'; folder.mkdir(parents=True)
            with gzip.open(folder/'trace.jsonl.gz','wt') as h:
                for r in records: h.write(json.dumps(r)+'\n')
            args=parser().parse_args(['--run',str(source),'--out',str(root/'profile'),
                                      '--steps','2','--repeats','1'])
            blocks=lambda m:list(m.model.transformer.blocks)
            with patch('experiments.multi_window.profile.load_model',return_value=(Model().eval(),Tokenizer())), \
                 patch('experiments.token_sleep.simulation.checked_blocks',blocks), \
                 patch('experiments.token_sleep.selective.checked_blocks',blocks), \
                 patch('torch.cuda.Event',Event),patch('torch.cuda.synchronize'), \
                 patch('torch.cuda.get_device_name',return_value='CPU fixture'),redirect_stdout(StringIO()):
                run(args)
            report=json.loads((root/'profile'/'profile.json').read_text())
            self.assertEqual(len(report['calls']),6)
            self.assertEqual(report['old_vs_new_max_probability_difference'],0.)
            self.assertEqual(report['old_vs_new_selection_changes'],0)
            self.assertEqual(report['timing']['updated_selective']['sparse']['calls'],1)


if __name__=='__main__': unittest.main()
