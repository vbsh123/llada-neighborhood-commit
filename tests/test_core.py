import unittest
from confidence_geography.core import groups, nearest, numeric_answer, select, step_metrics, token_category

def row(p, conf=.95):
    return dict(position=p, confidence=conf, eligible=True, special=False,
                previous_confidence=.4, delta_confidence=conf-.4)

class CoreTests(unittest.TestCase):
    def test_nonlocal_means_nearest_previous_commit_not_frontier(self):
        self.assertTrue(nearest(1, [7])['nonlocal'])
        self.assertFalse(nearest(8, [7])['nonlocal'])
        self.assertFalse(nearest(6, [7])['nonlocal'])
        self.assertIsNone(nearest(8, [])['nonlocal'])
        self.assertEqual(nearest(1, [7])['signed_distance'], -6)
    def test_batch_uses_nearest_anchor_and_tie_breaks_left(self):
        self.assertFalse(nearest(10, [0, 9])['nonlocal'])
        self.assertEqual(nearest(5, [3, 7])['nearest_previous_commit'], 3)
    def test_confident_islands_and_remote_count(self):
        m = step_metrics([row(p) for p in [0, 2, 3, 6]], None, [1], [.9])
        t = m['thresholds'][0]
        self.assertEqual(t['islands'], [[0, 0], [2, 3], [6, 6]])
        self.assertEqual(t['nonlocal_count'], 2)
        self.assertEqual(t['newly_above_count'], 4)
        self.assertEqual(groups([3, 2, 2, 6]), [[2, 3], [6]])
    def test_winner_removed_is_not_overtaking(self):
        self.assertEqual(step_metrics([row(3)], {'leader_position': 0}, [0], [.9])['leader_change_reason'], 'previous_winner_filled')
        self.assertEqual(step_metrics([row(0, .5), row(3)], {'leader_position': 0}, [2], [.9])['leader_change_reason'], 'overtook')
    def test_policy_eligibility_threshold_and_fallback(self):
        rows = [row(0, .8), row(1, .95), {**row(2, .99), 'eligible': False}]
        self.assertEqual([r['position'] for r in select(rows, 'threshold', 1, .9)], [1])
        self.assertEqual([r['position'] for r in select(rows, 'threshold', 1, .999)], [1])
        self.assertEqual([r['position'] for r in select(rows, 'left_to_right', 1, .9)], [0])
        self.assertEqual([r['position'] for r in select([row(2), row(1)], 'top1', 1, .9)], [1])
    def test_forced_jump_distinguished_from_remote_preference(self):
        m = step_metrics([row(0, .8), row(3, .95)], None, [1], [.9])
        self.assertTrue(m['nonlocal'])
        self.assertTrue(m['local_option_available'])
        self.assertEqual(m['uniform_choice_nonlocal_rate'], .5)
        m = step_metrics([row(3)], None, [1], [.9])
        self.assertTrue(m['nonlocal'])
        self.assertFalse(m['local_option_available'])

    def test_numeric_scoring_distinguishes_fallback(self):
        self.assertEqual(numeric_answer('reason 7 then #### 1,200.0'), ('1.2E+3', 'marked'))
        self.assertEqual(numeric_answer('It is 12.'), ('12', 'last_number'))
        self.assertEqual(numeric_answer('12', reference=True), (None, 'missing'))
    def test_token_categories(self):
        self.assertEqual(token_category(' 123'), 'number')
        self.assertEqual(token_category('\n'), 'whitespace')
        self.assertEqual(token_category('!'), 'punctuation_or_symbol')
        self.assertEqual(token_category('<eos>', True), 'special')

if __name__ == '__main__': unittest.main()
