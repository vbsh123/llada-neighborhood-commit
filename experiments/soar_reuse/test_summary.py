import io
import unittest

from run_probe import Probe


class ProbeSummaryTests(unittest.TestCase):
    def test_final_stop_filter_and_consistency_not_correctness(self):
        probe = Probe(io.StringIO(), [10])
        probe.batch_sizes = [2,1]
        def proposal(p, token, support, conflict):
            return {'position':p, 'token_id':token, 'donors_agree':True,
                    'best_parent_agrees':support,
                    'donors':[{'state_conflicts_with_best':[0] if conflict else []}]}
        probe.events = [
            {'step':0,'opportunities':[proposal(0,3,True,False),proposal(1,9,True,True),proposal(3,8,True,False)],
             'search_parents':2,'beam_width':2,'retained_width':1,'pruned':1,'duplicates':0},
            {'step':1,'opportunities':[proposal(0,3,False,False)],
             'search_parents':0,'beam_width':1,'retained_width':1,'pruned':1,'duplicates':0}]
        result = probe.summary([3,4,10,8],2)
        self.assertEqual(result['all_extra_position_events'],3)
        self.assertEqual(result['donor_consensus']['proposal_events'],3)
        self.assertEqual(result['donor_consensus']['unique_positions'],2)
        self.assertEqual(result['donor_consensus']['matches_final_output'],2)
        self.assertEqual(result['also_supported_by_best_parent']['proposal_events'],2)
        self.assertEqual(result['also_has_nonconflicting_donor_state']['proposal_events'],1)
        self.assertEqual(result['evaluated_branch_sequences'],3)
        self.assertEqual(result['model_calls'],2)

    def test_after_records_actual_survivors_and_donor_support(self):
        handle = io.StringIO();probe = Probe(handle,[])
        probe.mask_id=99
        probe.parents=[{'rows':{0:{'position':0,'token_id':3,'confidence':.8}},'search_trigger':True}]
        probe.candidates=[{'id':0,'parent':0,'state':[99,1]}, {'id':1,'parent':0,'state':[3,99]}]
        retained_entry=object();pruned_entry=object()
        probe.candidate_lookup={id(retained_entry):0,id(pruned_entry):1}
        probe.mixture=[]
        probe.after(0,[retained_entry,pruned_entry],[retained_entry,pruned_entry],[retained_entry])
        rows=probe.events[0]['opportunities']
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]['best_parent_agrees'])
        self.assertEqual(probe.events[0]['pruned'],1)
        self.assertIn('"retained_ids":[0]',handle.getvalue())


if __name__=='__main__':unittest.main()
