"""Execute the pinned decoder on tiny synthetic CPU logits, never model weights."""
import importlib.util
import io
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from checkout import DEFAULT_ROOT, verify_checkout
from hooks import instrument
from run_probe import Probe


@unittest.skipUnless(importlib.util.find_spec('torch') and (DEFAULT_ROOT/'.git').exists(),
                     'Requires PyTorch and the pinned local SOAR checkout')
class UpstreamExecutionTests(unittest.TestCase):
    def test_real_decoder_deduplication_and_parallel_collapse(self):
        import torch
        source = verify_checkout(DEFAULT_ROOT)
        # Decoder does not use these imported checkpoint-loading classes.
        unused_transformers = types.ModuleType('transformers')
        unused_transformers.AutoTokenizer = object
        unused_transformers.AutoModel = object
        for high_confidence in (False, True):
            with self.subTest(high_confidence=high_confidence):
                probe = Probe(io.StringIO(), {3})
                namespace = {'probe': probe}
                with patch.dict('sys.modules', {'transformers': unused_transformers}):
                    exec(compile(instrument(source.read_text()), str(source), 'exec'), namespace)
                class SyntheticLogits:
                    device = torch.device('cpu')
                    def __call__(self, x):
                        logits = torch.zeros((*x.shape, 4))
                        for position in range(x.shape[1]):
                            logits[:, position, position % 3] = 8. if high_confidence else 1.5
                        return types.SimpleNamespace(logits=logits)
                result = namespace['generate_soar'](
                    SyntheticLogits(), torch.tensor([[1]]), steps=4,
                    gen_length=4, block_length=4, mask_id=3, max_beam_size=2)
                self.assertEqual(result.shape, (1, 5))
                self.assertFalse((result[:, 1:] == 3).any())
                self.assertGreater(len(probe.lineage_steps), 0)
                self.assertTrue(all(step['retained'] for step in probe.lineage_steps))
                if not high_confidence:
                    self.assertTrue(any(e['beam_width'] == 2 for e in probe.events))
                else:
                    self.assertEqual(len(probe.lineage_steps), 1)
