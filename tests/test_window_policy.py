import unittest
from confidence_geography.core import select
from confidence_geography.run import parser


def row(p, confidence, eligible=True):
    return {'position':p,'confidence':confidence,'eligible':eligible,'token_id':p+100}


class WindowPolicyTests(unittest.TestCase):
    def test_neighbors_use_same_predictions_even_below_threshold(self):
        rows=[row(5,.99),row(4,.1),row(6,.2),row(7,.3),row(12,.98)]
        chosen=select(rows,'top1_window',1,.9)
        self.assertEqual([r['position'] for r in chosen],[5,4,6])
        self.assertEqual([r['token_id'] for r in chosen],[105,104,106])

    def test_missing_filled_and_outside_block_slots_not_replaced(self):
        rows=[row(5,.99),row(4,.9,False),row(7,.8),row(12,.98)]
        self.assertEqual([r['position'] for r in select(rows,'top1_window',1,.9)],[5])

    def test_left_edge_and_tie_break(self):
        rows=[row(2,.99),row(0,.99),row(1,.1)]
        self.assertEqual([r['position'] for r in select(rows,'top1_window',1,.9)],[0,1])

    def test_only_anchor_remaining_makes_progress(self):
        self.assertEqual(select([row(9,.2)],'top1_window',1,.9),[row(9,.2)])
        self.assertEqual(select([],'top1_window',1,.9),[])

    def test_cli_accepts_new_policy(self):
        self.assertEqual(parser().parse_args(['--out','unused','--policy','top1_window']).policy,'top1_window')


if __name__=='__main__':unittest.main()
