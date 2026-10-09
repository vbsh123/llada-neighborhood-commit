import json
from pathlib import Path
import tempfile
import unittest
from inputs import load_inputs, question_prompt, select_samples


class InputTests(unittest.TestCase):
    def test_independent_selection_is_repeatable_and_offsets_do_not_overlap(self):
        rows = [{'question': str(i), 'answer': f'#### {i}'} for i in range(10)]
        first = select_samples(rows, 3, 0, 1729)
        self.assertEqual(first, select_samples(rows, 3, 0, 1729))
        second = select_samples(rows, 3, 3, 1729)
        self.assertFalse({s['id'] for s in first} & {s['id'] for s in second})
        with self.assertRaises(ValueError):
            select_samples(rows, 20, 0, 1729)

    def test_offline_rows_and_resume_require_no_previous_run_or_download(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'data.jsonl'
            source.write_text(json.dumps({'question': 'Q', 'answer': '#### 7'})+'\n')
            config = {'data_jsonl': str(source), 'samples': 1, 'offset': 0, 'seed': 1729}
            samples, info = load_inputs(config)
            self.assertIn('sha256', info)
            saved = Path(folder)/'samples.jsonl'
            saved.write_text(''.join(json.dumps(s)+'\n' for s in samples))
            source.unlink()
            resumed, _ = load_inputs(config, saved)
            self.assertEqual(resumed, samples)

    def test_prompt_is_original_question_only_with_chat_marker(self):
        class Tokenizer:
            eos_token_id = 9
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                self.marker = add_generation_prompt
                return 'rendered'
            def encode(self, text, add_special_tokens):
                self.special = add_special_tokens
                return [1, 2]
            def get_vocab(self):
                return {'<|eot_id|>': 10}
        tokenizer = Tokenizer()
        header = question_prompt(tokenizer, 'Original question?', 99)
        self.assertEqual(tokenizer.messages, [{'role': 'user', 'content': 'Original question?'}])
        self.assertTrue(tokenizer.marker)
        self.assertFalse(tokenizer.special)
        self.assertEqual(header['stop_ids'], [9, 10])
        with self.assertRaisesRegex(ValueError, 'MASK'):
            question_prompt(tokenizer, 'Question', 2)


if __name__ == '__main__':
    unittest.main()
