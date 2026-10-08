import ast
import unittest
from hooks import instrument, reuse_opportunities


def candidate(cid, state, status='pruned'):
    return {'id': cid, 'state': state, 'status': status}


class ReuseTests(unittest.TestCase):
    def test_extra_tokens_and_conflicts_are_separate(self):
        candidates = [candidate(0,[1,99,99],'retained'), candidate(1,[2,3,99])]
        best, rows = reuse_opportunities(candidates,[0],99,set())
        self.assertEqual(best['id'],0)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['position'],1)
        self.assertEqual(rows[0]['token_id'],3)
        self.assertEqual(rows[0]['donors'][0]['state_conflicts_with_best'],[0])

    def test_donor_conflict_not_mislabeled_consensus(self):
        candidates = [candidate(0,[99],'retained'),candidate(1,[3]),candidate(2,[4])]
        _,rows = reuse_opportunities(candidates,[0],99,set())
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['donors_agree'])
        self.assertIsNone(rows[0]['token_id'])

    def test_duplicate_donor_and_special_tokens_excluded(self):
        candidates = [candidate(0,[99,99],'retained'),candidate(1,[3,99],'duplicate'),candidate(2,[99,10])]
        self.assertEqual(reuse_opportunities(candidates,[0],99,{10})[1],[])

    def test_token_already_in_another_surviving_branch_is_not_lost(self):
        candidates = [candidate(0,[99,99],'retained'),candidate(1,[3,99],'retained'),candidate(2,[3,4])]
        _,rows = reuse_opportunities(candidates,[0,1],99,set())
        self.assertEqual([r['position'] for r in rows],[1])

    def test_multiple_agreeing_donors_count_once_per_position(self):
        candidates = [candidate(0,[99],'retained'),candidate(1,[3]),candidate(2,[3])]
        _,rows = reuse_opportunities(candidates,[0],99,set())
        self.assertEqual(len(rows),1)
        self.assertEqual(len(rows[0]['donors']),2)
        self.assertTrue(rows[0]['donors_agree'])


class InstrumentationTests(unittest.TestCase):
    def test_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unexpected upstream'):
            instrument('def generate_soar(): pass')

    def test_only_observer_statements_are_added(self):
        source = '''
def generate_soar():
    for global_step in range(2):
        new_beam_candidates = []
        for beam_idx in range(2):
            if beam_idx == 0:
                new_beam_candidates.append(1)
            elif beam_idx == 1:
                new_beam_candidates.append(2)
            elif beam_idx == 2:
                new_beam_candidates.append(3)
            else:
                new_beam_candidates.append(4)
        best_seq, best_score, best_block, best_records = beam[0]
    return best_seq
'''
        tree = instrument(source,check_hash=False)
        class StripObservers(ast.NodeTransformer):
            def visit_Expr(self,node):
                value=node.value
                if (isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute)
                    and isinstance(value.func.value,ast.Name) and value.func.value.id=='probe'):
                    return None
                return node
        stripped=StripObservers().visit(tree)
        self.assertEqual(ast.dump(stripped),ast.dump(ast.parse(source)))


if __name__=='__main__':unittest.main()
