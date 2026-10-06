import json
from pathlib import Path
import tempfile
import unittest
from confidence_geography.inspect_window_failures import inspect_window_failures


class WindowFailureTests(unittest.TestCase):
    def write(self,root,sid,correct,question='q'):
        path=root/'samples'/sid;path.mkdir(parents=True)
        (path/'result.json').write_text(json.dumps({'sample_id':sid,'correct_numeric':correct,
            'question':question,'reference':'#### 1','answer':'<script>bad</script>',
            'predicted_answer':'1' if correct else '2','gold_answer':'1'}))

    def test_filters_requested_direction_and_preserves_reverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for sid,x,y in [('a',True,False),('b',False,True),('c',True,True),('d',False,False)]:
                self.write(p/'four',sid,x);self.write(p/'three',sid,y)
            r=inspect_window_failures(p/'four',p/'three',p/'out')
            self.assertEqual([c['sample_id'] for c in r['four_correct_three_wrong']],['a'])
            self.assertEqual([c['sample_id'] for c in r['three_correct_four_wrong']],['b'])
            self.assertNotIn('<script>bad</script>',(p/'out/index.html').read_text())

    def test_mismatched_question_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.write(p/'four','a',True);self.write(p/'three','a',False,'different')
            with self.assertRaisesRegex(ValueError,'Question/reference mismatch'):
                inspect_window_failures(p/'four',p/'three',p/'out')


if __name__=='__main__':unittest.main()
