import unittest
from convergence import measure_convergence


def parent(i, token, support=.2):
    return {'id': i, 'rows': {0: {'token_id': token, 'confidence': .95,
                                'cross_support': {'7': support, '8': 1-support}}}}


class ConvergenceTests(unittest.TestCase):
    def test_support_without_top1_then_convergence_and_commit(self):
        steps = [
            {'step': 0, 'parents': [parent(0, 7), parent(1, 8)],
             'retained': [{'parent': 1, 'state': [99]}]},
            {'step': 1, 'parents': [parent(0, 7, .96)],
             'retained': [{'parent': 0, 'state': [7]}]}]
        events, summary = measure_convergence(steps, [7], 1, {99})
        target = next(e for e in events if e['recipient_parent'] == 1)
        self.assertEqual(target['recipient_support'], .2)
        self.assertEqual(target['first_top1_delay'], 1)
        self.assertEqual(target['first_commit_delay'], 1)
        self.assertTrue(target['on_final_lineage'])
        self.assertTrue(target['matches_final_output'])
        self.assertEqual(target['max_later_masked_support'], .96)
        sweep = next(s for s in summary['threshold_sweep']
                     if s['donor_threshold'] == .9 and s['recipient_support_threshold'] == .2)
        self.assertEqual(sweep['observations'], 2)
        self.assertEqual(sweep['final_lineage_observations'], 1)

    def test_commit_now_does_not_invent_later_masked_top1(self):
        steps = [{'step': 0, 'parents': [parent(0, 7), parent(1, 8)],
                  'retained': [{'parent': 1, 'state': [7]}]},
                 {'step': 1, 'parents': [{'id': 0, 'rows': {}}],
                  'retained': [{'parent': 0, 'state': [7]}]}]
        events, _ = measure_convergence(steps, [7], 1, {99})
        target = next(e for e in events if e['recipient_parent'] == 1)
        self.assertEqual(target['first_commit_delay'], 0)
        self.assertIsNone(target['first_top1_delay'])
        self.assertIsNone(target['max_later_masked_support'])

    def test_pruned_recipient_censored_and_not_final_match(self):
        steps = [{'step': 0, 'parents': [parent(0, 7), parent(1, 8)],
                  'retained': [{'parent': 0, 'state': [7]}]}]
        events, _ = measure_convergence(steps, [7], 1, {99})
        target = next(e for e in events if e['recipient_parent'] == 1)
        self.assertTrue(target['recipient_lineage_extinct'])
        self.assertFalse(target['on_final_lineage'])
        self.assertIsNone(target['matches_final_output'])
        self.assertIsNone(target['first_commit_delay'])

    def test_follow_all_descendants_despite_beam_reordering(self):
        steps = [{'step': 0, 'parents': [parent(0, 7), parent(1, 8)],
                  'retained': [{'parent': 1, 'state': [99]}, {'parent': 1, 'state': [99]}]},
                 {'step': 1, 'parents': [parent(0, 8), parent(1, 7)],
                  'retained': [{'parent': 1, 'state': [7]}, {'parent': 0, 'state': [8]}]}]
        events, _ = measure_convergence(steps, [7], 1, {99})
        target = next(e for e in events if e['recipient_parent'] == 1)
        self.assertEqual(target['first_top1_delay'], 1)
        self.assertEqual(target['first_commit_delay'], 1)
        self.assertTrue(target['on_final_lineage'])

    def test_special_tokens_and_post_stop_positions_excluded(self):
        steps = [{'step': 0, 'parents': [parent(0, 7), parent(1, 8)],
                  'retained': [{'parent': 0, 'state': [7]}]}]
        self.assertEqual(measure_convergence(steps, [7], 0, set())[0], [])
        events, _ = measure_convergence(steps, [7], 1, {7, 8})
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
