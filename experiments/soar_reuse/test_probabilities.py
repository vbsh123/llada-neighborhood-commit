"""Small CPU tensor check; skipped in environments without PyTorch."""
import importlib.util
import io
import unittest
from run_probe import Probe


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
class ProbabilityTests(unittest.TestCase):
    def test_non_top1_support_and_historical_token_support(self):
        import torch
        probe = Probe(io.StringIO(), {3})
        beam = [(torch.tensor([[0, 3, 3]]), 0., 0, []) for _ in range(2)]
        logits = torch.tensor([[[0., 0., 0., 0.], [0., 3., 0., -10.], [0., 3., 0., -10.]],
                               [[0., 0., 0., 0.], [0., 1., 3., -10.], [0., 1., 3., -10.]]])
        prob = logits.softmax(-1)
        confidence, predictions = prob.max(-1)
        probe.before(0, beam, logits, predictions, confidence, 1, 2, 3, .95)
        self.assertEqual(probe.parents[0]['rows'][0]['token_id'], 1)
        self.assertEqual(probe.parents[1]['rows'][0]['token_id'], 2)
        self.assertAlmostEqual(probe.parents[1]['rows'][0]['cross_support']['1'],
                               float(prob[1, 1, 1]), places=6)
        self.assertAlmostEqual(probe.token_probability(1, 0, 1), float(prob[1, 1, 1]), places=6)
        # Previously proposed token 1 remains scored even after all top1s change.
        next_logits = torch.zeros_like(logits)
        next_logits[:, :, 2] = 3.
        next_prob = next_logits.softmax(-1)
        next_conf, next_pred = next_prob.max(-1)
        probe.before(1, beam, next_logits, next_pred, next_conf, 1, 2, 3, .95)
        self.assertIn('1', probe.parents[1]['rows'][0]['cross_support'])
        self.assertGreater(probe.parents[0]['rows'][0]['entropy'], 0)
