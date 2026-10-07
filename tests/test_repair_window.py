import contextlib
import copy
import gzip
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from confidence_geography.repair_window import repair_candidates, repair_window, repaired_result


class Tokenizer:
    all_special_ids = [99]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return ''.join(str(i) for i in ids if not (skip_special_tokens and i == 99))

    def convert_ids_to_tokens(self, token_id):
        return str(token_id)


def fixture():
    original = {'sample_id': 'a', 'final_ids': [1, 2, 3, 4, 99, 6],
                'answer_token_length': 4, 'first_stop_position': 4,
                'first_stop_step': 1, 'steps': 2, 'elapsed_seconds': 10.,
                'answer': '1234', 'reference': '#### 1234', 'correct_numeric': True,
                'token_dictionary': {}, 'word_map': ['stale']}
    header = {'type': 'header', 'prompt_ids': [42],
              'config': {'policy': 'top1_window', 'mask_id': 100, 'top_k': 5}}
    def step(n, anchor, positions):
        return {'type': 'step', 'step': n, 'selection': {'anchor_position': anchor},
                'commit_positions': positions,
                'positions': [{'position': p, 'token_id': original['final_ids'][p],
                               'confidence': .5, 'special': p == 4} for p in positions]}
    return original, [header, step(0, 1, [1, 0, 2, 3]), step(1, 5, [5, 4])]


class RepairTests(unittest.TestCase):
    def test_only_forced_nonspecial_before_stop_strict_threshold(self):
        original, records = fixture()
        records[1]['positions'][2]['confidence'] = .75
        _, candidates = repair_candidates(records, original, .75)
        self.assertEqual([c['position'] for c in candidates], [0, 3])
        self.assertEqual([c['offset_from_anchor'] for c in candidates], [-1, 2])
        # A post-stop ordinary forced neighbor is excluded too.
        records[2]['selection']['anchor_position'] = 4
        self.assertNotIn(5, [c['position'] for c in repair_candidates(records, original, .75)[1]])

    def test_three_position_version_and_no_candidates(self):
        original, records = fixture()
        records[1]['commit_positions'].remove(3)
        self.assertEqual([c['position'] for c in repair_candidates(records, original, .75)[1]], [0, 2])
        self.assertEqual(repair_candidates(records, original, 0)[1], [])

    def test_bad_threshold_or_duplicate_commit_rejected(self):
        original, records = fixture()
        with self.assertRaises(ValueError):
            repair_candidates(records, original, 1.1)
        records.append(records[1])
        with self.assertRaisesRegex(ValueError, 'more than once'):
            repair_candidates(records, original, .75)

    def test_scoring_keeps_original_boundary_even_with_new_special(self):
        original, _ = fixture()
        new = repaired_result(original, [1, 99, 3, 4, 99, 6], Tokenizer(),
                              [{'token_id': 99}], 2., 1.)
        self.assertEqual(new['answer'], '134')
        self.assertFalse(new['correct_numeric'])
        self.assertEqual(new['first_stop_position'], 4)
        self.assertEqual(new['answer_token_length'], 4)
        self.assertEqual(new['steps'], 3)
        self.assertEqual(new['elapsed_seconds'], 12.)
        self.assertNotIn('word_map', new)
        self.assertEqual(original['final_ids'], [1, 2, 3, 4, 99, 6])

    def write_source(self, root, original, records):
        folder = root / 'samples' / 'a'
        folder.mkdir(parents=True)
        (folder / 'result.json').write_text(json.dumps(original))
        with gzip.open(folder / 'trace.jsonl.gz', 'wt') as handle:
            for record in records:
                handle.write(json.dumps(record) + '\n')

    def test_zero_candidates_never_load_model_and_resume(self):
        original, records = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_source(root / 'source', original, records)
            with patch('confidence_geography.repair_window.load_model') as load:
                result = repair_window(root / 'source', root / 'out', 0)
                repair_window(root / 'source', root / 'out', 0)
                load.assert_not_called()
            self.assertEqual(result['extra_repair_forwards'], 0)
            self.assertEqual(result['repaired_accuracy'], 1.)

    def test_all_candidates_masked_together_one_forward_no_acceptance_gate(self):
        original, records = fixture()
        inputs = []
        class Tensor:
            is_cuda = False
            def __init__(self, ids):
                self.ids = ids
            def __getitem__(self, key):
                return self
        def model(x, use_cache):
            self.assertFalse(use_cache)
            inputs.append(copy.deepcopy(x.ids))
            return types.SimpleNamespace(logits=Tensor([]))
        model.device = 'fake'
        fake_torch = types.SimpleNamespace(tensor=lambda ids, **kw: Tensor(ids),
                                           inference_mode=contextlib.nullcontext)
        # Low-confidence repaired predictions are still committed, exactly once.
        rows = [{'position': p, 'token_id': 8, 'confidence': .1, 'entropy': 3.}
                for p in [0, 2, 3]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_source(root / 'source', original, records)
            with patch.dict('sys.modules', {'torch': fake_torch}), \
                 patch('confidence_geography.repair_window.load_model', return_value=(model, Tokenizer())), \
                 patch('confidence_geography.repair_window.score_logits', return_value=rows) as score:
                summary = repair_window(root / 'source', root / 'out')
                saved = json.loads((root / 'out/samples/a/result.json').read_text())
                # Resuming must not repeat the forward.
                repair_window(root / 'source', root / 'out')
            self.assertEqual(inputs, [[[42, 100, 2, 100, 100, 99, 6]]])
            score.assert_called_once()
            self.assertEqual(saved['final_ids'], [8, 2, 8, 8, 99, 6])
            self.assertEqual(summary['extra_repair_forwards'], 1)
            self.assertEqual(summary['changed_tokens'], 3)


if __name__ == '__main__':
    unittest.main()
