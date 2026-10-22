"""CPU checks for measurements, task scoring, and attention interventions."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.audio_drift.inputs import read_manifest, score
from experiments.audio_drift.protocol import anchors_for, conditions, work_estimate
from experiments.audio_drift.stability import timing

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import torch
    from experiments.audio_drift.decode import Prepared, decode, distribution_change, schedule
    from experiments.audio_drift.hooks import AttentionProbe
    from experiments.audio_drift.metrics import DriftRecorder, drift, token_groups


class InputTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_priority_balance_after40_crossing_and_bounds(self):
        from experiments.audio_drift.attention_priority import BalanceRamp
        ramp=BalanceRamp(40,1.)
        self.assertEqual(ramp.update(20,1,9)['phase'],'before_start')
        self.assertEqual(ramp.update(40,1,9)['phase'],'waiting_for_semantic_dominance')
        self.assertIsNone(ramp.cutoff)
        self.assertEqual(ramp.update(41,3,1)['semantic_weight'],1.)
        self.assertAlmostEqual(ramp.update(42,2,1)['semantic_weight'],2/3)
        cross=ramp.update(43,1,1)
        self.assertEqual(cross['cutoff'],43);self.assertEqual(cross['semantic_weight'],0.)
        self.assertAlmostEqual(ramp.update(44,1,3)['text_weight'],.5)
        bounce=ramp.update(45,3,1)
        self.assertEqual(bounce['semantic_weight'],0.);self.assertAlmostEqual(bounce['text_weight'],.5)

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_crossing_configuration_and_source_order(self):
        from experiments.audio_drift.crossing_pruning import crossing_conditions,dependency_order,resolve_control
        a,b=crossing_conditions()
        self.assertEqual(dependency_order([b,a]),[a,b])
        with self.assertRaises(ValueError):dependency_order([b])
        for kw in [dict(fractions=[.5,.25]),dict(fractions=[float('nan')]),dict(interval=0),dict(patience=0)]:
            with self.assertRaises(ValueError):crossing_conditions(**kw)
        empty=resolve_control(b,[dict(condition=a['condition'],accounting=[dict(pruning=dict(selections=[],trigger_checks=[]))])])
        self.assertEqual(empty['pruning']['monitor_steps'],[])
        self.assertGreater(empty['pruning']['stages'][0]['start'],10**6)

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_delayed_pruning_matrix_pairs_every_strategy_across_scopes(self):
        from experiments.audio_drift.pruning import pruning_conditions
        from experiments.audio_drift.protocol import is_identity
        specs = pruning_conditions()
        self.assertEqual(len(specs), 20)
        self.assertEqual(len({s['condition'] for s in specs}), 20)
        groups = {}
        for spec in specs:
            p = spec['pruning']
            key = (p['start'], p['selection'], spec['schedule'][0]['refresh_every'])
            groups.setdefault(key, set()).add(p['scope'])
            self.assertFalse(is_identity(spec))
            self.assertEqual(spec['schedule'][0]['refresh_every'], 1)
        self.assertTrue(all(g == {'acoustic', 'semantic', 'audio', 'prompt', 'prefix'} for g in groups.values()))
        for kw in [dict(starts=[0]), dict(fractions=[1.1]), dict(fractions=[float('nan')]), dict(layer_start=-1)]:
            with self.assertRaises(ValueError): pruning_conditions(**kw)

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_staged_matrix_and_duration_pilot_selection(self):
        from experiments.audio_drift.staged_pruning import staged_conditions
        from experiments.audio_drift.select_pilot import select
        specs=staged_conditions()
        self.assertEqual(len(specs),12)
        names={s['condition'] for s in specs}
        self.assertTrue(all(s['paired_condition'] in names for s in specs))
        self.assertTrue(all(s['schedule']==[{'group':'prefix','refresh_every':1}] for s in specs))
        self.assertEqual({len(s['pruning']['stages']) for s in specs},{1,2,3})
        self.assertEqual(select([{'id':1,'audio_seconds':10},{'id':2,'audio_seconds':20}],1,15)[0]['id'],2)
        with self.assertRaises(ValueError):select([],2,15)
        for kw in [dict(fractions=[1.1]),dict(deep_layer=0),dict(acoustic_start=0),dict(later_start=1)]:
            with self.assertRaises(ValueError):staged_conditions(**kw)

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_single_pool_budget_has_no_shared_63_cap(self):
        from experiments.audio_drift.pruning import LatePruning
        prepared=Prepared(torch.ones((1,76),dtype=torch.long),torch.randn(1,274,4),0,64,31)
        q,k=torch.randn(1,1,354,4),torch.randn(1,1,354,4)
        for scope,fraction,count in [('semantic',.75,157),('prefix',.5,175),('acoustic',1.,64)]:
            spec=dict(start=1,fraction=fraction,layer_start=0,queries=2,seed=1729,scope=scope,selection='attention')
            engine=LatePruning(prepared,1,spec)
            keep=engine.select(0,1,q,k,None)
            self.assertEqual(len(keep),354-count)
            self.assertEqual(engine.records[0]['drop_budget'],count)
            self.assertEqual(engine.records[0]['budget_basis'],'scope_fraction')

    def test_refresh_math_includes_initialization_and_short_runs(self):
        from experiments.audio_drift.budget import refresh_count, row_budget
        from types import SimpleNamespace
        self.assertEqual([refresh_count(64, r) for r in [8, 16, 32, 0]], [8, 4, 2, 1])
        self.assertEqual(refresh_count(16, 16), 1)
        self.assertEqual(refresh_count(17, 16), 2)
        prepared = SimpleNamespace(answer_start=6, group_positions=lambda g: {'acoustic': [1, 2],
            'prompt': [0, 5], 'semantic': [3, 4], 'prefix': list(range(6))}[g])
        spec = conditions(['freeze'], triple_refresh=['16:8:16'])[-1]
        budget = row_budget(prepared, 4, 64, spec, 2)
        # Four answer rows every step, prompt eight times, each audio group four times.
        self.assertEqual(budget['computed_rows_by_layer'], [288, 288])
        self.assertEqual(budget['dense_token_layers'], 1280)

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_low_attention_episodes_and_future_reactivation_are_not_conflated(self):
        from experiments.audio_drift.attention_recurrence import low_episode_anchors, reactivation_records
        low = np.asarray([[1], [1], [1], [0], [1], [1]], dtype=bool)
        times, tokens = low_episode_anchors(low, np.zeros_like(low), 2)
        self.assertEqual(times.tolist(), [1, 5])
        self.assertEqual(tokens.tolist(), [0, 0])
        mean = np.full((8, 1), .1); mean[6] = 3.
        rows = reactivation_records(mean, mean, history=2, horizons=(2, 4, 6, 8))
        lookup = {r['horizon']: r for r in rows if r['anchor_decode_quarter'] == -1}
        self.assertEqual(lookup[4]['eligible_episodes'], 1)
        self.assertEqual(lookup[4]['reactivated_episodes'], 0)
        self.assertEqual(lookup[6]['reactivated_episodes'], 1)
        self.assertEqual(lookup[6]['first_high_delay_histogram'], {'5': 1})
        self.assertEqual(lookup[8]['eligible_episodes'], 0)
        self.assertEqual(lookup[8]['right_censored_episodes'], 1)
        self.assertIsNone(lookup[8]['probability'])

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_mean_attention_can_hide_future_head_spikes(self):
        from experiments.audio_drift.attention_recurrence import reactivation_records
        mean = np.full((8, 1), .1); peak = mean.copy(); peak[4] = 3.
        average = reactivation_records(mean, peak, history=1, horizons=(4,), high_metric='mean')[0]
        spike = reactivation_records(mean, peak, history=1, horizons=(4,), high_metric='peak')[0]
        self.assertEqual(average['reactivated_episodes'], 0)
        self.assertEqual(spike['reactivated_episodes'], 1)
        # A spike on ANOTHER token must not reactivate the low-attention token.
        mean = np.full((8, 2), .1); mean[:, 1] = 3.
        row = reactivation_records(mean, mean, history=1, horizons=(4,))[0]
        self.assertEqual(row['eligible_episodes'], 1)
        self.assertEqual(row['reactivated_episodes'], 0)

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_high_to_low_has_separate_direction_and_requires_peak_to_fall(self):
        from experiments.audio_drift.attention_recurrence import reactivation_records
        mean = np.full((8, 1), 3.); mean[6] = .1
        rows = reactivation_records(mean, mean, history=2, horizons=(6, 8), direction='high_to_low')
        selected = {r['horizon']: r for r in rows if r['anchor_decode_quarter'] == -1}
        self.assertEqual(selected[6]['transitioned_episodes'], 1)
        self.assertEqual(selected[6]['first_target_delay_histogram'], {'5': 1})
        self.assertIsNone(selected[6]['reactivated_episodes'])
        self.assertEqual(selected[8]['right_censored_episodes'], 1)
        # Mean can fall while a head still attends strongly: peak-to-low must NOT fire.
        peak = np.full((8, 1), 3.)
        row = reactivation_records(mean, peak, history=2, horizons=(6,),
                                   high_metric='peak', direction='high_to_low')[0]
        self.assertEqual(row['transitioned_episodes'], 0)

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_diagnostic_js_correlations_and_refresh_phases(self):
        from experiments.audio_drift.attention_dynamics import js_distance, correlations, eos_forward_count
        from experiments.audio_drift.refresh_profile import phase_name
        p = torch.tensor([[1., 0.], [.5, .5]])
        torch.testing.assert_close(js_distance(p, p), torch.zeros(2))
        self.assertAlmostEqual(float(js_distance(p[:1], p[:1].flip(-1))), __import__('math').log(2), places=6)
        self.assertIsNone(correlations([1, 1, 1], [1, 2, 3])['spearman'])
        self.assertAlmostEqual(correlations([1, 2, 3], [6, 4, 2])['spearman'], -1)
        spec = conditions(['freeze'], triple_refresh=['32:8:32'])[-1]
        self.assertEqual(phase_name(0, spec), 'initialization')
        self.assertEqual(phase_name(8, spec), 'prompt')
        self.assertEqual(phase_name(32, spec), 'acoustic+prompt+semantic')
        self.assertEqual(phase_name(1, spec), 'cache_hit')
        events = [{'committed_positions': [1]}, {'committed_positions': [0]}, {'committed_positions': [2]}]
        self.assertEqual(eos_forward_count(events, [4, 30, 9], 30), 2)

    def test_final_numeric_answer_scoring(self):
        for text, target in [("3 + 15 = 18.\n#### 18", "18"),
                             ("Final answer: $1,200.00", "1200"),
                             ("-0.00", "0"), ("#### -2.50", "-2.5"),
                             ("18", "18.0"), ("#### 8\n#### 18", "18")]:
            with self.subTest(text=text):
                self.assertTrue(score(text, target, "numeric")["correct"])
        for text in ["There are 18 apples", "Compute 3 + 15", "#### 1,20", "#### NaN",
                     "#### 18 or 19", "#### 18\nMore calculation 4", "#### 1/0"]:
            with self.subTest(text=text):
                self.assertTrue(score(text, "18", "numeric")["parse_failed"])
        self.assertFalse(score("#### 19", "18", "numeric")["correct"])
        with self.assertRaises(ValueError):
            score("18", "not a number", "numeric")

    def test_three_group_refresh_and_benchmark_only(self):
        from types import SimpleNamespace
        from experiments.audio_drift.run import intervention_specs, interventions
        args = SimpleNamespace(interventions=["freeze"], freeze_groups=["prefix"],
                               refresh_intervals=[1, 4, 8], triple_refresh=["8:4:4", "16:4:4", "8:8:4"],
                               condition_names=["freeze_prefix_r1", "freeze_prefix_r4", "freeze_prefix_r8",
                                                "freeze_triple_a8_p4_s4", "freeze_triple_a16_p4_s4",
                                                "freeze_triple_a8_p8_s4"], skip_dense_interventions=True)
        specs = intervention_specs(args)
        self.assertEqual(len(specs), 6)
        triple = next(s for s in specs if s["condition"] == "freeze_triple_a8_p4_s4")
        self.assertEqual(triple["schedule"], [{"group": "acoustic", "refresh_every": 8},
                                              {"group": "prompt", "refresh_every": 4},
                                              {"group": "semantic", "refresh_every": 4}])
        self.assertEqual(interventions(None, None, None, None, args, None, None), [])
        for bad in ["8:4", "8:-1:4"]:
            with self.assertRaises(ValueError):
                conditions(["freeze"], triple_refresh=[bad])

    @unittest.skipUnless(HAS_TORCH, "audio dependencies unavailable")
    def test_spoken_math_reservoir_and_duration_filter(self):
        import io
        import soundfile as sf
        from experiments.audio_drift.prepare_gsm import select_rows
        def rows():
            for i in range(12):
                data = io.BytesIO()
                sf.write(data, np.zeros(1600 if i < 10 else 3200), 1600, format="WAV")
                yield "data/test-0.parquet", i, {"audio": {"bytes": data.getvalue()}, "answer": str(i)}
        a, counts = select_rows(rows(), 3, 1729, 0, 1.5)
        b, _ = select_rows(rows(), 3, 1729, 0, 1.5)
        self.assertEqual([r['source_row'] for r in a], [r['source_row'] for r in b])
        self.assertEqual(counts, {"scanned": 12, "eligible": 10})
        self.assertEqual(len({r['source_row'] for r in a}), 3)
        self.assertTrue(all(r['audio_seconds'] == 1 for r in a))
        with self.assertRaises(ValueError):
            select_rows(rows(), 11, 1729, 0, 1.5)

    def test_reasoning_report_pairs_accuracy_and_latency(self):
        from experiments.audio_drift.reasoning_report import summarize
        def result(i, baseline, cached):
            return {"id": str(i), "target": "18", "score": score(baseline, "18", "numeric"),
                    "answer_has_no_eos": False,
                    "benchmark": {"baseline_median_seconds": 4, "baseline_forwards": 8,
                      "policies": [{"condition": "test", "score": score(cached, "18", "numeric"),
                                    "median_seconds": 2, "accounting": [{"forwards": 8}],
                                    "same_forward_count_as_baseline": True, "answer_has_no_eos": False}]}}
        rows, summary = summarize([result(0, "18", "19"), result(1, "19", "18"),
                                   result(2, "18", "No final answer")])
        item = summary["test"]
        self.assertEqual(item["samples"], 3)
        self.assertEqual(item["correct_to_wrong"], 2)
        self.assertEqual(item["wrong_to_correct"], 1)
        self.assertEqual(item["parse_failures"], 1)
        self.assertEqual(item["decode_speedup_ratio_of_sums"], 2)
        self.assertEqual(item["same_forward_count_samples"], 3)
        paired = result(0, "#### Answer: 18", "#### 18")
        paired["fixed_input_replay"] = {"forwards": 2, "baseline_median_seconds": 3,
                                      "policies": [{"condition": "test", "median_seconds": 2}]}
        rows, summary = summarize([paired])
        self.assertFalse(rows[0]["baseline_strict_correct"])
        self.assertTrue(rows[0]["baseline_correct"])
        self.assertEqual(summary['test']['fixed_input_replay']['forward_speedup_ratio_of_sums'], 1.5)
        self.assertEqual(summary['test']['fixed_input_replay']['forwards_total'], 2)

    @unittest.skipUnless(HAS_TORCH, "audio dependencies unavailable")
    def test_spoken_math_preparation_from_local_parquet(self):
        import io
        import soundfile as sf
        import pyarrow as pa
        import pyarrow.parquet as pq
        from types import SimpleNamespace
        from experiments.audio_drift.prepare_gsm import main, PROMPT
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = io.BytesIO()
            sf.write(data, np.zeros(1600), 1600, format="WAV")
            shard = directory / "test.parquet"
            pq.write_table(pa.Table.from_pylist([{"audio": {"bytes": data.getvalue(), "path": "audio.wav"},
                                                 "answer": str(i)} for i in range(5)]), shard)
            manifest = directory / "pilot.jsonl"
            api = SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="pinned"),
                                  list_repo_files=lambda *a, **k: ["data/test-00000-of-00001.parquet"])
            with patch("sys.argv", ["prepare_gsm", "--out", str(manifest), "--samples", "2"]), \
                 patch("huggingface_hub.HfApi", return_value=api), \
                 patch("huggingface_hub.hf_hub_download", return_value=str(shard)):
                main()
            rows = read_manifest(manifest)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['metric'] == 'numeric' and r['prompt'] == PROMPT for r in rows))
            self.assertTrue(all(r['source_revision'] == 'pinned' for r in rows))
            provenance = json.loads(manifest.with_suffix('.provenance.json').read_text())
            self.assertEqual(provenance['scanned'], 5)
            self.assertEqual(provenance['eligible'], 5)

    def test_numeric_labels_preserve_strict_audit(self):
        for text in ["#### Answer: 2", "#### <final numerical answer> 2"]:
            value = score(text, "2", "numeric")
            self.assertTrue(value["correct"])
            self.assertTrue(value["format_normalized"])
            self.assertFalse(value["strict_correct"])
            self.assertIsNone(value["strict_prediction"])
        for text in ["#### Answer: 2 or 6", "#### <final numerical answer> 2 apples",
                     "Answer: 2", "Intermediate answer 2"]:
            self.assertTrue(score(text, "2", "numeric")["parse_failed"])
        value = score("#### 6", "2", "numeric")
        self.assertFalse(value["correct"])
        self.assertFalse(value["format_normalized"])

    def test_acoustic_sweep_matrix(self):
        from types import SimpleNamespace
        from experiments.audio_drift.run import intervention_specs
        args = SimpleNamespace(interventions=["freeze"], freeze_groups=["prefix"],
                               refresh_intervals=[1, 8, 16], triple_refresh=["16:8:8", "32:8:8", "0:8:8", "16:8:16"],
                               condition_names=["freeze_prefix_r1", "freeze_prefix_r8", "freeze_prefix_r16",
                               "freeze_triple_a16_p8_s8", "freeze_triple_a32_p8_s8",
                               "freeze_triple_a0_p8_s8", "freeze_triple_a16_p8_s16"])
        specs = intervention_specs(args)
        self.assertEqual(len(specs), 7)
        permanent = next(s for s in specs if s["condition"] == "freeze_triple_a0_p8_s8")
        self.assertEqual(permanent["schedule"][0], {"group": "acoustic", "refresh_every": 0})
        from experiments.audio_drift.decode import schedule
        plan = schedule(512, 32, 256)
        self.assertEqual(len(plan), 256)
        self.assertEqual(sum(c for _, _, c in plan), 512)
        self.assertTrue(all(c == 2 for _, _, c in plan))

    def test_depth_summary_includes_layer_zero_and_excludes_late_steps(self):
        import numpy as np
        from experiments.audio_drift.paper_plots import depth_summary, GROUPS
        values = np.ones((32, 4))
        values[:, 0] = 0; values[0, 1:3] = 0; values[:, 3] = 100
        matrices = {('previous', g): values.copy() for g in GROUPS}
        result = depth_summary(matrices, list(range(32)), 2)
        self.assertTrue(result['all_32_layers_observed'])
        for g in GROUPS:
            self.assertAlmostEqual(result['active_mean_all_measured_layers'][g], 31/32)
            self.assertEqual(result['active_mean_by_layer'][g]['0'], 0)
        subset = {('previous', g): values[[7,15,23,31]] for g in GROUPS}
        self.assertFalse(depth_summary(subset, [7,15,23,31], 2)['all_32_layers_observed'])
        matrices['previous', GROUPS[0]][1,1] = np.nan
        with self.assertRaises(ValueError):
            depth_summary(matrices, list(range(32)), 2)

    def test_depth_variant_expansion_and_validation(self):
        from experiments.audio_drift.protocol import with_depth_variants, layer_intervals, is_identity
        base = conditions(['freeze'], ['prefix'], [1,8])
        result = with_depth_variants(base, [8,4,2,1])
        identity = next(r for r in result if r['condition']=='freeze_prefix_r1')
        self.assertTrue(is_identity(identity))
        self.assertFalse(any(r['condition'].startswith('freeze_prefix_r1_depth') for r in result))
        depth = next(r for r in result if r['condition']=='freeze_prefix_r8_depth8x4x2x1')
        self.assertEqual(layer_intervals(depth['schedule'][0],32), (64,)*8+(32,)*8+(16,)*8+(8,)*8)
        self.assertEqual(depth['flat_condition'],'freeze_prefix_r8')
        with self.assertRaises(ValueError): with_depth_variants(base,[3,2,1])
        with self.assertRaises(ValueError): layer_intervals(depth['schedule'][0],3)
        with self.assertRaises(ValueError): layer_intervals({'refresh_every':2,'refresh_by_layer_group':[2,4]},32)
        self.assertFalse(is_identity({'mode':'freeze','schedule':[{'group':'prefix','refresh_every':1,
                                                                  'refresh_by_layer_group':[4,1]}]}))

    def test_night_policy_matrix_has_six_pairs_and_identity(self):
        from types import SimpleNamespace
        from experiments.audio_drift.run import intervention_specs
        args = SimpleNamespace(interventions=['freeze'], freeze_groups=['prefix'], refresh_intervals=[1,8],
            triple_refresh=['16:8:16','16:8:8','16:8:4','8:8:4','16:16:4'],depth_multipliers=[8,4,2,1],
            condition_names=['freeze_prefix_r1','freeze_prefix_r8','freeze_triple_a16_p8_s16',
                             'freeze_triple_a16_p8_s8','freeze_triple_a16_p8_s4',
                             'freeze_triple_a8_p8_s4','freeze_triple_a16_p16_s4'])
        policies=intervention_specs(args)
        self.assertEqual(len(policies),13)
        self.assertEqual(len({p['condition'] for p in policies}),13)
        self.assertEqual(sum('flat_condition' in p for p in policies),6)
        from experiments.audio_drift.benchmark import policy_specs
        args.shuffle_policy_order=True;args.seed=1729;args.benchmark_conditions=[]
        order=policy_specs(args,'sample-one')
        self.assertEqual([p['condition'] for p in order],[p['condition'] for p in policy_specs(args,'sample-one')])
        self.assertEqual(order[0]['condition'],'freeze_prefix_r1')
        self.assertEqual({p['condition'] for p in order},{p['condition'] for p in policies})
        with self.assertRaises(ValueError):policy_specs(args)
        for p in policies:
            if 'flat_condition' in p:
                self.assertTrue(any(q['condition']==p['flat_condition'] for q in policies))

    def test_marker_comma_scoring_preserves_strict_grade(self):
        for text in ['####,120','####, 120','#### Answer: ,120']:
            value=score(text,'120','numeric')
            self.assertTrue(value['correct']);self.assertFalse(value['strict_correct'])
        for text in ['####,','####,120 or 20','Reasoning says 120','####, profit is 120']:
            self.assertTrue(score(text,'120','numeric')['parse_failed'])

    @unittest.skipUnless(HAS_TORCH, 'audio dependencies unavailable')
    def test_combined_manifest_keeps_development_and_excludes_it_from_fresh(self):
        import io
        import soundfile as sf
        import pyarrow as pa
        import pyarrow.parquet as pq
        from unittest.mock import Mock
        from experiments.audio_drift.prepare_gsm import main, DATASET
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);buffer=io.BytesIO()
            sf.write(buffer,np.zeros(1600),1600,format='WAV')
            wave=directory/'base.wav';wave.write_bytes(buffer.getvalue())
            base=directory/'base.jsonl'
            base.write_text(''.join(json.dumps({'id':f'gsm8k-speech-{i:05}', 'audio':'base.wav',
                'task':'gsm8k_speech','prompt':'original instruction','target':str(i),'metric':'numeric',
                'source_dataset':DATASET,'source_revision':'pinned','source_file':'data/test-0.parquet','source_row':i})+'\n' for i in [1,3]))
            shard=directory/'test.parquet'
            pq.write_table(pa.Table.from_pylist([{'audio':{'bytes':buffer.getvalue(),'path':'audio.wav'},
                                                  'answer':str(i)} for i in range(12)]),shard)
            api=Mock();api.dataset_info.return_value.sha='pinned'
            api.list_repo_files.return_value=['data/test-0.parquet']
            manifest=directory/'combined.jsonl'
            with patch('sys.argv',['prepare_gsm','--out',str(manifest),'--samples','6','--base-manifest',str(base)]), \
                 patch('huggingface_hub.HfApi',return_value=api), \
                 patch('huggingface_hub.hf_hub_download',return_value=str(shard)):
                main()
            rows=read_manifest(manifest);dev=[r for r in rows if r['cohort']=='development'];fresh=[r for r in rows if r['cohort']=='fresh']
            self.assertEqual(len(dev),2);self.assertEqual(len(fresh),4)
            self.assertFalse({r['id'] for r in dev}&{r['id'] for r in fresh})
            self.assertEqual([r['cohort'] for r in rows[:4]],['development','fresh','development','fresh'])
            self.assertTrue(all(r['prompt']=='original instruction' for r in dev))
            api.dataset_info.assert_called_once_with(DATASET,revision='pinned')
            provenance=json.loads(manifest.with_suffix('.provenance.json').read_text())
            self.assertEqual(provenance['excluded_source_indices'],[1,3])
            self.assertEqual(provenance['eligible'],10)

    def test_long_pilot_selects_only_named_conditions(self):
        from types import SimpleNamespace
        from experiments.audio_drift.run import intervention_specs
        args = SimpleNamespace(interventions=["freeze"], freeze_groups=["acoustic", "semantic", "both"],
                               refresh_intervals=[1, 4, 8], mixed_refresh=["8:2", "2:8"],
                               condition_names=["freeze_acoustic", "freeze_semantic", "freeze", "freeze_both_r1",
                                                "freeze_both_r4", "freeze_both_r8", "freeze_mixed_a8_s2", "freeze_mixed_a2_s8"])
        specs = intervention_specs(args)
        self.assertEqual({s["condition"] for s in specs}, set(args.condition_names))
        self.assertEqual(work_estimate(128, [0], specs)["branches"], 8)
        args.condition_names.append("typo")
        with self.assertRaisesRegex(ValueError, "typo"):
            intervention_specs(args)

    def test_full_condition_matrix_and_work_budget(self):
        specs = conditions(["freeze", "remove", "text_only"], ["acoustic", "semantic", "both"],
                           [1, 2, 4, 8], ["8:2", "2:8"])
        self.assertEqual(len(specs), 19)
        self.assertEqual(len({s["condition"] for s in specs}), 19)
        mixed = next(s for s in specs if s["condition"] == "freeze_mixed_a8_s2")
        self.assertEqual(mixed["schedule"], [{"group": "acoustic", "refresh_every": 8},
                                             {"group": "semantic", "refresh_every": 2}])
        estimate = work_estimate(128, [0, .25, .5, .75], specs)
        self.assertEqual(estimate["branches"], 76)
        self.assertEqual(estimate["continuation_forwards"], 6023)
        self.assertEqual(anchors_for(4, [0, 0, .999]), [0])
        for invalid in [["8"], ["-1:2"]]:
            with self.assertRaises(ValueError):
                conditions(["freeze"], mixed_refresh=invalid)
        with self.assertRaises(ValueError):
            conditions(["freeze"], refresh_intervals=[0])

    def test_stability_distinguishes_temporary_dip_from_stable_tail(self):
        values = [(0, .1), (1, .01), (2, .01), (3, .01), (4, .1),
                  (5, .01), (6, .01), (7, .01)]
        result = timing(values, .02, window=3)
        self.assertEqual(result["first_sustained_step"], 1)
        self.assertEqual(result["stable_through_end_step"], 5)
        self.assertEqual(result["last_violation_step"], 4)
        self.assertIsNone(timing([(0, .01), (2, .01), (3, .01)], .02, 3)["first_sustained_step"])
        self.assertIsNone(timing([(0, .1), (1, .01)], .02, 3)["stable_through_end_step"])
        with self.assertRaises(ValueError):
            timing([(0, float("nan"))], .02)

    def test_reference_scoring(self):
        self.assertEqual(score("Hello, world!", "HELLO WORLD", "wer")["edits"], 0)
        self.assertEqual(score("a x c d", "a b c", "wer")["edits"], 2)
        self.assertTrue(score("Answer: (B).", "B", "mcq")["correct"])
        self.assertFalse(score("Because the answer is B", "B", "mcq")["correct"])
        self.assertTrue(score(" HAPPY. ", "happy", "exact")["correct"])

    def test_manifest_resolves_audio_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.wav").write_bytes(b"placeholder")
            row = dict(id="clip", audio="clip.wav", task="asr", prompt="Transcribe")
            manifest = root / "inputs.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            loaded = read_manifest(manifest)
            self.assertEqual(loaded[0]["audio"], str(root / "clip.wav"))
            self.assertEqual(len(loaded[0]["audio_sha256"]), 64)
            manifest.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_manifest(manifest)


@unittest.skipUnless(HAS_TORCH, "Optional CPU torch dependency is absent")
class MetricTests(unittest.TestCase):
    def test_pairwise_cosine_matches_token_mean_and_handles_zero_vectors(self):
        from experiments.audio_drift.pairwise import PairwiseRecorder
        labels = ["prompt", "audio_acoustic", "audio_semantic", "prompt"]
        values = [torch.tensor([[1., 0.], [0., 0.], [1., 1.], [0., 1.]]),
                  torch.tensor([[0., 1.], [1., 0.], [1., 0.], [0., 1.]]),
                  torch.tensor([[-1., 0.], [0., 0.], [0., 1.], [1., 0.]])]
        with tempfile.TemporaryDirectory() as directory:
            recorder = PairwiseRecorder(directory, labels, [0, 1, 2], [0], "cpu")
            for step, value in enumerate(values):
                recorder.observe(0, "key", value, {"step": step})  # ignored
                recorder.observe(0, "hidden", value, {"step": step})
            self.assertFalse(recorder.history)  # vectors released after last selected step
            recorder.close()
            data = json.loads((Path(directory) / "pairwise_cosine.json").read_text())
            self.assertTrue(data["complete"])
            matrices = {r["group"]: np.array(r["cosine_similarity"]) for r in data["records"]}
            self.assertAlmostEqual(matrices["prompt"][0, 1], .5, places=6)
            self.assertAlmostEqual(matrices["prompt"][0, 2], -.5, places=6)
            self.assertEqual(matrices["audio_acoustic"][0, 1], 0.)
            self.assertEqual(matrices["audio_acoustic"][0, 2], 1.)
            for matrix in matrices.values():
                np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)
                np.testing.assert_allclose(np.diag(matrix), 1.)
            self.assertNotIn("tensors", data)

    def test_local_parquet_reader_reads_only_selected_split_and_closes_files(self):
        from experiments.audio_drift.prepare import local_parquet_rows
        from types import SimpleNamespace
        import pyarrow as pa
        import pyarrow.parquet as pq
        files = ["clean/train.100/0000.parquet", "clean/test/0001.parquet",
                 "other/test/0000.parquet", "clean/test/0000.parquet", "README.md"]
        api = SimpleNamespace(list_repo_files=lambda *a, **k: files)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.parquet"
            pq.write_table(pa.Table.from_pylist([{"id": "x", "text": "HELLO",
                                                 "audio": {"bytes": b"audio", "path": "x.flac"}}]), path)
            with patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download:
                rows, selected = local_parquet_rows("repo", "clean", "test", "fixed", api)
            self.assertEqual(selected, ["clean/test/0000.parquet", "clean/test/0001.parquet"])
            self.assertEqual([r["audio"]["bytes"] for r in rows], [b"audio", b"audio"])
            self.assertEqual(download.call_count, 2)
            self.assertTrue(all(c.kwargs["revision"] == "fixed" for c in download.call_args_list))

    def test_duration_quotas_filter_boundaries_and_interleave(self):
        from experiments.audio_drift.prepare import main
        from types import SimpleNamespace
        import soundfile
        # A clip at the shared 20s boundary belongs only to the longer bin.
        examples = [(9, 0), (20, 1), (30, 2), (25, 3), (10, 4), (19, 5), (15, 6)]
        rows = [{"id": str(i), "speaker_id": i, "text": "HELLO", "audio": {"bytes": str(s).encode()}}
                for s, i in examples]
        class Dataset:
            def cast_column(self, *args): return self
            def __iter__(self): return iter(rows)
            def __len__(self): return len(rows)
        datasets = SimpleNamespace(Audio=lambda **kwargs: object(), load_dataset=lambda *a, **k: Dataset())
        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="fixed")))
        def read_audio(source, **kwargs):
            return np.zeros(int(source.read()) * 10, dtype=np.float32), 10
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "balanced.jsonl"
            argv = ["prepare", "--out", str(manifest), "--samples", "4", "--min-seconds", "10",
                    "--reader", "local-parquet", "--duration-bins", "10:20:2", "20:30:2"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"datasets": datasets, "huggingface_hub": hub}), \
                    patch.object(soundfile, "read", side_effect=read_audio), \
                    patch("experiments.audio_drift.prepare.local_parquet_rows", return_value=(rows, ["clean/test/x.parquet"])):
                main()
            selected = read_manifest(manifest)
            self.assertEqual([r["audio_seconds"] for r in selected], [10, 20, 19, 30])
            self.assertEqual([r["duration_bin"] for r in selected], ["10-20", "20-30"] * 2)
            provenance = json.loads(manifest.with_suffix(".provenance.json").read_text())
            self.assertEqual([b["selected"] for b in provenance["duration_bins"]], [2, 2])
            self.assertEqual(provenance["reader"], "local-parquet")
            for invalid in [["10:25:2", "20:30:2"], ["10:20:3"]]:
                with patch.object(sys, "argv", ["prepare", "--out", str(manifest), "--samples", "4",
                                                "--duration-bins", *invalid]):
                    with self.assertRaises(ValueError): main()

    def test_streaming_preparation_records_selection_and_speaker_cap(self):
        from experiments.audio_drift.prepare import main
        from types import SimpleNamespace
        import soundfile
        rows = [{"id": name, "speaker_id": speaker, "text": "HELLO",
                 "audio": {"bytes": name.encode()}}
                for name, speaker in [("long", 1), ("first", 1), ("duplicate_speaker", 1), ("second", 2)]]
        calls = {}
        class Dataset:
            def cast_column(self, *args):
                return self
            def shuffle(self, **kwargs):
                calls["shuffle"] = kwargs
                return self
            def __iter__(self):
                return iter(rows)
        def load_dataset(*args, **kwargs):
            calls["load"] = kwargs
            return Dataset()
        datasets = SimpleNamespace(Audio=lambda **kwargs: object(), load_dataset=load_dataset)
        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **k: SimpleNamespace(sha="immutable")))
        def read_audio(source, **kwargs):
            seconds = 31 if source.read() == b"long" else 1
            return np.zeros(16000 * seconds, dtype=np.float32), 16000
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "test.jsonl"
            argv = ["prepare", "--out", str(manifest), "--samples", "2", "--streaming",
                    "--shuffle-seed", "1729", "--max-per-speaker", "1"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"datasets": datasets, "huggingface_hub": hub}), \
                    patch.object(soundfile, "read", side_effect=read_audio):
                main()
            result = read_manifest(manifest)
            self.assertEqual([r["id"] for r in result], ["first", "second"])
            self.assertEqual(calls["shuffle"], {"seed": 1729, "buffer_size": 1000})
            self.assertTrue(calls["load"]["streaming"])
            provenance = json.loads(manifest.with_suffix(".provenance.json").read_text())
            self.assertEqual(provenance["revision"], "immutable")
            self.assertEqual(provenance["skipped_duration"], 1)
            self.assertEqual(provenance["speaker_counts"], {"1": 1, "2": 1})

    def test_summary_keeps_conditions_separate_and_excludes_finished_answers(self):
        from experiments.audio_drift.run import aggregate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (words, base_edits, changed_edits, finished) in enumerate(
                    [(17, 1, 3, False), (11, 9, 9, True)]):
                sample = root / "samples" / str(index)
                sample.mkdir(parents=True)
                baseline = {"metric": "wer", "edits": base_edits, "reference_words": words,
                            "wer": base_edits / words}
                branches = []
                for condition in ["freeze_acoustic_r2", "freeze_semantic_r2"]:
                    changed = {"metric": "wer", "edits": changed_edits, "reference_words": words,
                               "wer": changed_edits / words}
                    branches.append({"mode": "freeze", "condition": condition, "score": changed,
                                     "anchor_step_fraction": .5, "same_final_tokens": False,
                                     "baseline_answer_already_revealed": finished})
                (sample / "result.json").write_text(json.dumps({"task": "asr", "score": baseline,
                                                               "interventions": branches}))
            aggregate(root)
            summary = json.loads((root / "summary.json").read_text())["tasks"]["asr"]["conditions"]
            self.assertEqual(len(summary), 3)
            condition = summary["freeze_acoustic_r2@0.500"]
            self.assertAlmostEqual(condition["corpus_wer"], 12 / 28)
            self.assertEqual(condition["already_revealed_samples"], 1)
            self.assertEqual(condition["with_content_remaining"]["samples"], 1)
            self.assertAlmostEqual(condition["with_content_remaining"]["corpus_wer_delta"], 2 / 17)

    def test_magnitude_and_direction_are_distinct(self):
        current = torch.tensor([[2., 0.], [0., 1.], [0., 0.]])
        previous = torch.tensor([[1., 0.], [1., 0.], [0., 0.]])
        values = drift(current, previous)
        torch.testing.assert_close(values["relative_l2"], torch.tensor([1., 2 ** .5, 0.]))
        torch.testing.assert_close(values["cosine_similarity"], torch.tensor([1., 0., 1.]))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            drift(torch.tensor([[float("nan")]]), torch.ones(1, 1))

    def test_stable_controls_exclude_identity_changes(self):
        labels = ["audio_semantic", "audio_acoustic", "prompt", "answer", "answer", "answer"]
        groups = token_groups(labels, torch.tensor([-1, -1, 3, 9, 4, 5]),
                               torch.tensor([-1, -1, 3, 9, 9, 5]), 9)
        self.assertEqual(groups["answer_masked"].nonzero().flatten().tolist(), [3])
        self.assertEqual(groups["answer_changed"].nonzero().flatten().tolist(), [4])
        self.assertEqual(groups["answer_decoded"].nonzero().flatten().tolist(), [5])

    def test_reference_age_and_block_reset_and_scalar_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DriftRecorder(directory, ["audio_semantic", "prompt"], 9, per_token=True)
            for step in range(3):
                recorder.begin({"step": step, "step_in_block": step % 2}, torch.tensor([-1, 2]))
                for kind in ("hidden", "key", "value"):
                    recorder.observe(0, kind, torch.ones(2, 3) * (step + 1))
                recorder.end([0])
            recorder.close()
            rows = [json.loads(line) for line in (Path(directory) / "drift.jsonl").read_text().splitlines()]
            row = next(r for r in rows if r["step"] == 2 and r["reference"] == "previous" and r["group"] == "audio" and r["tensor"] == "key")
            self.assertEqual(row["age"], 1)
            self.assertAlmostEqual(row["metrics"]["relative_l2"]["median"], .5)
            row = next(r for r in rows if r["step"] == 2 and r["reference"] == "block_start")
            self.assertEqual(row["age"], 0)
            data = np.load(Path(directory) / "token_metrics" / "0002.npz", allow_pickle=False)
            self.assertEqual(data["l000_key_previous_relative_l2"].shape, (2,))
            self.assertFalse((Path(directory) / "snapshots").exists())

    def test_distribution_change_and_schedule(self):
        values = torch.tensor([[[1., 0.], [0., 2.]]])
        same = distribution_change(values, values, torch.tensor([[True, True]]))
        self.assertEqual(same["kl_mean"], 0.)
        changed = distribution_change(values, values.flip(-1), torch.tensor([[False, True]]))
        self.assertEqual(changed["top1_change_rate"], 1.)
        self.assertGreater(changed["kl_mean"], 0)
        self.assertEqual([count for _, _, count in schedule(8, 4, 4)], [2, 2, 2, 2])
        with self.assertRaises(ValueError):
            schedule(8, 3, 4)


@unittest.skipUnless(HAS_TORCH and os.environ.get("DIFFA_ROOT"),
                     "Set DIFFA_ROOT to test the pinned upstream transformer on CPU")
class UpstreamTests(unittest.TestCase):
    def setUp(self):
        from experiments.audio_drift.checkout import verify_checkout
        source = verify_checkout(os.environ["DIFFA_ROOT"])
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        from src.fast_dllm.model.configuration_llada import LLaDAConfig
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
        torch.manual_seed(10)
        config = LLaDAConfig(d_model=16, n_heads=2, n_kv_heads=2, n_layers=2,
                             mlp_hidden_size=32, vocab_size=32, embedding_size=32,
                             block_type="llama", block_group_size=1, rope=True,
                             activation_type="silu",
                             max_sequence_length=64, flash_attention=False,
                             weight_tying=False, mask_token_id=31, eos_token_id=30,
                             attention_dropout=0., residual_dropout=0., embedding_dropout=0.)
        self.llm = LLaDAModelLM(config, init_params=True).eval().to(torch.bfloat16)
        # Avoid predicting MASK in a random tiny model without altering the
        # decoder's selection rule, which must match upstream exactly.
        output = torch.nn.Linear(16, 32, bias=True).to(torch.bfloat16)
        with torch.no_grad():
            output.bias[31] = -100
        self.llm.model.transformer.ff_out = output
        self.prepared = Prepared(torch.tensor([[1, 2, 3]]),
                                 torch.randn(1, 2, 16).to(torch.bfloat16), 1, 1, 31)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_online_attention_scores_zero_priority_and_priority_selection(self):
        from types import SimpleNamespace
        from experiments.audio_drift.attention_scores import OnlineAttentionScores
        from experiments.audio_drift.attention_priority import AttentionPriority
        from experiments.audio_drift.decode import forward_logits,commit
        self.llm.float()
        x=torch.tensor([[1,2,3,31,31,31,31]])
        expected=forward_logits(self.llm,self.prepared,x)
        with OnlineAttentionScores(self.llm,self.prepared,special_ids=[30,31]) as probe:
            probe.begin(x,0,2,30)
            actual=forward_logits(self.llm,self.prepared,x)
            key,mass=probe.summaries()
            self.assertEqual(probe.query_offsets.tolist(),[0,1])
            self.assertEqual(tuple(mass.shape),(2,4));self.assertEqual(len(probe.layer_mass),2)
            self.assertEqual(mass[:,3].sum(),0.)
            self.assertTrue(bool((key>=0).all()));self.assertTrue(bool((mass.sum(-1)<=1.00001).all()))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        tokenizer=SimpleNamespace(eos_token_id=30,all_special_ids=[30,31])
        baseline,_,_=decode(self.llm,self.prepared,4,2,4)
        with AttentionPriority(self.llm,self.prepared,tokenizer,alpha=0,start=0) as control:
            identity,_,_=decode(self.llm,self.prepared,4,2,4,priority=control)
        self.assertTrue(torch.equal(identity,baseline))
        self.assertEqual(len(control.records),4)
        self.assertTrue(all(all(v==0 for v in r['priority_bonus']) for r in control.records))
        # A priority can change which position commits without changing the argmax token.
        logits=torch.tensor([[[0.,2.,0.],[0.,1.9,0.],[0.,1.,0.],[0.,1.,0.]]])
        plain=x.clone();boosted=x.clone()
        _,p=commit(plain,logits,self.prepared,0,2,1)
        _,q=commit(boosted,logits,self.prepared,0,2,1,torch.tensor([[0.,.2,0.,0.]]))
        self.assertEqual(p,[0]);self.assertEqual(q,[1]);self.assertEqual(int(boosted[0,4]),1)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_sensitivity_checkpoints_measurements_and_priority_report(self):
        from types import SimpleNamespace
        from experiments.audio_drift.removal_sensitivity import collect_sensitivity,prediction_change,checkpoint_states
        from experiments.audio_drift.attention_priority import run_priority_study
        from experiments.audio_drift.attention_study_report import analyze
        from experiments.audio_drift.run import parser,write_json
        self.prepared=Prepared(self.prepared.prompt,torch.randn(1,4,16).to(torch.bfloat16),1,2,31)
        self.llm.float()
        final,states,events=decode(self.llm,self.prepared,4,2,4)
        args=parser().parse_args(['--manifest','unused','--out','unused','--no-drift',
            '--length','4','--block-length','2','--steps','4','--sensitivity-layer','1',
            '--sensitivity-budgets','1','2','--priority-strengths','.02','--priority-start','0'])
        tokenizer=SimpleNamespace(eos_token_id=30,all_special_ids=[30,31],decode=lambda ids,**kw:'#### 1')
        points,reason=checkpoint_states(states,self.prepared,[7,8,9,10],[30,31],30,[.25,.5,.75])
        self.assertEqual([p['step'] for p in points],[1,2,3]);self.assertIsNone(reason)
        identical=prediction_change(torch.tensor([[1.,2.]]),torch.tensor([[1.,2.]]))
        self.assertEqual(identical['kl_mean'],0.);self.assertEqual(identical['top1_change_rate'],0.)
        with tempfile.TemporaryDirectory() as directory,patch.object(torch.cuda,'synchronize'):
            root=Path(directory);work=root/'samples/00000';work.mkdir(parents=True)
            summary=collect_sensitivity(self.llm,self.prepared,states,events,[7,8,9,10],tokenizer,args,work)
            self.assertEqual(summary['measurements'],48)
            rows=[json.loads(r) for r in (work/'removal_sensitivity.jsonl').read_text().splitlines()]
            self.assertTrue(all(len(r['removed_positions'])==r['budget'] for r in rows))
            for group in ('acoustic','semantic','prompt','prefix'):
                same=[r for r in rows if r['group']==group and r['budget']==1 and r['selection']=='fixed_early']
                self.assertTrue(all(r['removed_positions']==same[0]['removed_positions'] for r in same))
            result=run_priority_study(self.llm,self.prepared,final,args,tokenizer,dict(metric='numeric',target='1'),work)
            self.assertEqual(result['policies'],4)
            write_json(work/'result.json',dict(id='tiny',score=dict(correct=True)))
            # Plotting is not installed in the minimal CPU test environment.
            (work/'removal_sensitivity.jsonl').rename(work/'sensitivity.saved')
            sensitivity,priority=analyze(root)
            self.assertEqual(len(priority),4)
            self.assertTrue((root/'analysis/attention_priority_trajectories.csv').exists())

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_crossing_patience_progression_and_uncapped_budgets(self):
        from types import SimpleNamespace
        from experiments.audio_drift.crossing_pruning import CrossingPruning,crossing_conditions
        groups=dict(acoustic=list(range(100)),semantic=list(range(100,300)),prompt=list(range(300,380)))
        prepared=SimpleNamespace(group_positions=lambda g:groups[g])
        engine=CrossingPruning.__new__(CrossingPruning)
        engine.prepared=prepared;engine.spec=crossing_conditions(patience=2)[0]['pruning']
        engine.stages=engine.spec['stages'];engine.selected={};engine.position_ids=tuple(range(400));engine.checks=[]
        engine.streaks=dict(acoustic=0,semantic=0);positions=torch.arange(400)
        scores=torch.zeros(400);scores[:100]=.0001;scores[100:300]=.0005;scores[300:380]=.01
        for step,expected in [(8,[]),(16,[0,3]),(24,[]),(32,[1,4]),(40,[]),(48,[2,5])]:
            engine.step=step
            pending=engine.pending_stages(25)
            self.assertTrue(engine.should_score(25,pending))
            chosen=engine.choose_pending(scores,positions,pending)
            self.assertEqual(chosen,expected)
            for i in chosen:engine.selected[i]=()
        self.assertEqual(sum(engine.drop_budget(s) for s in engine.stages if s['group']=='semantic'),150)
        self.assertEqual(engine.drop_budget(dict(group='semantic',fraction=.75)),150)
        self.assertFalse(engine.should_score(25,[]))
        # A single failed crossing resets patience; groups trigger independently.
        engine.selected={};engine.streaks=dict(acoustic=0,semantic=0)
        engine.step=8;engine.choose_pending(scores,positions,engine.pending_stages(25))
        scores[:100]=.1;engine.step=16
        self.assertEqual(engine.choose_pending(scores,positions,engine.pending_stages(25)),[3])
        self.assertEqual(engine.streaks['acoustic'],0)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_crossing_real_compaction_pair_replay_and_dense_reference(self):
        from types import SimpleNamespace
        from experiments.audio_drift.crossing_pruning import CrossingPruning,crossing_conditions,resolve_control
        from experiments.audio_drift.compact import policy_context
        from experiments.audio_drift.decode import forward_logits
        from experiments.audio_drift.hooks import blocks_of
        from experiments.audio_drift.run import parser
        from experiments.audio_drift.benchmark import benchmark
        from experiments.audio_drift.replay import benchmark_replay
        self.prepared=Prepared(self.prepared.prompt,torch.randn(1,4,16).to(torch.bfloat16),1,2,31)
        self.llm.float()
        inputs=[torch.tensor([[1,2,3,31,31,31,31]]) for _ in range(4)]
        a,b=crossing_conditions([.5,1.],layer=1,start=1,interval=1,patience=1,queries=3)
        original_choose=CrossingPruning.choose_pending
        def guaranteed_crossing(engine,scores,positions,pending):
            forced=torch.zeros_like(scores)
            prompt=torch.tensor(engine.prepared.group_positions('prompt'))
            local=torch.searchsorted(positions,prompt)
            # Prompt positions survive in the audio source; generic is scheduled.
            if not engine.spec.get('resolved_control'):forced[local]=1.
            return original_choose(engine,scores if engine.spec.get('resolved_control') else forced,positions,pending)
        completed=[]
        with patch.object(CrossingPruning,'choose_pending',guaranteed_crossing):
            for spec in [a,b]:
                spec=resolve_control(spec,completed)
                with policy_context(self.llm,self.prepared,spec) as engine:
                    actual=[forward_logits(self.llm,self.prepared,x) for x in inputs]
                    counts=engine.accounting();picks=engine.records
                self.assertEqual(sum(p['drop_budget'] for p in picks),4)
                self.assertEqual([p['step'] for p in picks],[1,1,2,2])
                completed.append(dict(condition=spec['condition'],accounting=[counts]))
                # Dense model with identical removed KV columns is the numerical reference.
                original_methods=[];current=[0]
                try:
                    for layer,block in enumerate(blocks_of(self.llm)):
                        method=block._scaled_dot_product_attention
                        existed='_scaled_dot_product_attention' in block.__dict__
                        original_methods.append((block,method,existed))
                        def dense(q,k,v,_layer=layer,_method=method,**kwargs):
                            removed={p for r in picks if r['step']<=current[0] and r['layer']<=_layer for p in r['dropped_positions']}
                            keep=torch.tensor([i for i in range(11) if i not in removed])
                            return _method(q,k.index_select(-2,keep),v.index_select(-2,keep),**kwargs)
                        block._scaled_dot_product_attention=dense
                    for step,x in enumerate(inputs):
                        current[0]=step
                        torch.testing.assert_close(actual[step],forward_logits(self.llm,self.prepared,x),rtol=2e-5,atol=2e-5)
                finally:
                    for block,method,existed in original_methods:
                        if existed:block._scaled_dot_product_attention=method
                        else:delattr(block,'_scaled_dot_product_attention')
            self.assertEqual(completed[0]['accounting'][0]['computed_rows_by_layer'],completed[1]['accounting'][0]['computed_rows_by_layer'])
            args=parser().parse_args(['--manifest','unused','--out','unused','--interventions','--crossing-prune',
                '--crossing-fractions','.5','1','--crossing-layer','1','--crossing-start','1',
                '--crossing-interval','1','--crossing-patience','1','--prune-queries','3',
                '--skip-dense-interventions','--no-drift','--length','4','--block-length','2','--steps','4',
                '--benchmark-repeats','2','--replay-repeats','2','--shuffle-policy-order'])
            final,states,_=decode(self.llm,self.prepared,4,2,4)
            tokenizer=SimpleNamespace(eos_token_id=30,decode=lambda ids,**kw:'#### 1')
            with tempfile.TemporaryDirectory() as directory,patch.object(torch.cuda,'synchronize'), \
                    patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_allocated',return_value=0):
                work=Path(directory)
                bench=benchmark(self.llm,self.prepared,final,args,tokenizer,dict(id='crossing',metric='numeric',target='1'),work)
                replay=benchmark_replay(self.llm,self.prepared,states,bench['baseline_forwards'],args,work,sample_id='crossing')
                self.assertEqual(len(replay['policies']),2)
                self.assertTrue(bench['policies'][1]['pruning']['resolved_control'])
                self.assertEqual(len(bench['policies'][0]['accounting'][0]['pruning']['trigger_checks']),2)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_compact_stages_match_dense_removal_without_banks(self):
        from experiments.audio_drift.staged_pruning import staged_conditions, CompactStagedPruning
        from experiments.audio_drift.decode import forward_logits
        from experiments.audio_drift.hooks import blocks_of
        self.prepared=Prepared(self.prepared.prompt,torch.randn(1,4,16).to(torch.bfloat16),1,2,31)
        inputs=[torch.tensor([[1,2,3,31,31,31,31]]) for _ in range(4)]
        inputs[2][0,3]=5;inputs[3][0,4]=6
        for dtype in [torch.float32,torch.bfloat16]:
            self.llm.to(dtype)
            specs=staged_conditions([.5,1.],1,2,1,3)
            self.assertEqual(len(specs),12)
            for spec in specs:
                with CompactStagedPruning(self.llm,self.prepared,spec['pruning']) as engine:
                    actual=[forward_logits(self.llm,self.prepared,x) for x in inputs]
                    counts=engine.accounting()
                    self.assertFalse(hasattr(engine,'banks'))
                    picks=engine.records
                    self.assertEqual(len(picks),len(spec['pruning']['stages']))
                    self.assertTrue(all(p['drop_budget']==int(len(self.prepared.group_positions(p['group']))*spec['pruning']['fraction']) for p in picks))
                    self.assertTrue(all(p<self.prepared.answer_start for r in picks for p in r['dropped_positions']))
                    self.assertGreater(counts['skipped_token_update_fraction'],0)
                    if spec['pruning']['scope']=='asp' and spec['pruning']['fraction']==1.:
                        self.assertEqual(engine.events[-1]['layer_computed_rows'][-1],4)
                step=[-1];originals=[]
                try:
                    for layer,block in enumerate(blocks_of(self.llm)):
                        original=block._scaled_dot_product_attention;existed='_scaled_dot_product_attention' in block.__dict__
                        originals.append((block,original,existed))
                        def dense(q,k,v,_layer=layer,_original=original,**kwargs):
                            removed={p for r in picks if r['step']<=step[0] and r['layer']<=_layer for p in r['dropped_positions']}
                            keep=torch.tensor([i for i in range(11) if i not in removed])
                            return _original(q,k.index_select(-2,keep),v.index_select(-2,keep),**kwargs)
                        block._scaled_dot_product_attention=dense
                    reference=[]
                    for i,x in enumerate(inputs):
                        step[0]=i;reference.append(forward_logits(self.llm,self.prepared,x))
                finally:
                    for block,original,existed in originals:
                        if existed:block._scaled_dot_product_attention=original
                        else:delattr(block,'_scaled_dot_product_attention')
                for a,b in zip(actual,reference):
                    torch.testing.assert_close(a,b,rtol=.04 if dtype==torch.bfloat16 else 2e-5,atol=.04 if dtype==torch.bfloat16 else 2e-5)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_staged_benchmark_replay_reports_and_cpu_operator_capture(self):
        from types import SimpleNamespace
        from experiments.audio_drift.run import parser,write_json
        from experiments.audio_drift.benchmark import benchmark
        from experiments.audio_drift.replay import benchmark_replay
        from experiments.audio_drift.pruning_report import analyze
        from experiments.audio_drift.staged_profile import profile_staged
        self.prepared=Prepared(self.prepared.prompt,torch.randn(1,4,16).to(torch.bfloat16),1,2,31)
        self.llm.float()
        args=parser().parse_args(['--manifest','unused','--out','unused','--interventions','--staged-prune',
            '--staged-fractions','0.5','0.75','--acoustic-start','1','--later-start','2','--deep-layer','1',
            '--skip-dense-interventions','--no-drift','--length','4','--block-length','2','--steps','4',
            '--benchmark-repeats','1','--replay-repeats','1'])
        final,states,_=decode(self.llm,self.prepared,4,2,4)
        tokenizer=SimpleNamespace(eos_token_id=30,decode=lambda ids,**kw:'#### 1')
        actual_profiler=torch.profiler.profile
        def cpu_profile(**kw):
            kw['activities']=[torch.profiler.ProfilerActivity.CPU]
            return actual_profiler(**kw)
        with tempfile.TemporaryDirectory() as directory,patch.object(torch.cuda,'synchronize'), \
                patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_allocated',return_value=0):
            root=Path(directory);work=root/'samples/00000';work.mkdir(parents=True)
            bench=benchmark(self.llm,self.prepared,final,args,tokenizer,{'id':'tiny','metric':'numeric','target':'1'},work)
            replay=benchmark_replay(self.llm,self.prepared,states,bench['baseline_forwards'],args,work,sample_id='tiny')
            self.assertEqual(len(bench['policies']),12)
            self.assertEqual(len(replay['policies']),12)
            self.assertTrue(all(p['accounting'][0]['pruning']['selections'][-1]['anchor_progress'] for p in bench['policies']))
            write_json(work/'result.json',{'id':'tiny','score':{'correct':True},'benchmark':bench})
            rows=analyze(root)
            self.assertEqual(len(rows),12)
            self.assertEqual(max(r['activated_stages'] for r in rows),3)
            with patch.object(torch.profiler,'profile',side_effect=cpu_profile):
                profile_staged(self.llm,self.prepared,states,4,args,work,'tiny')
            trace=json.loads((work/'pruning_traces/index.json').read_text())
            self.assertTrue(trace['captures'])
            labels={s['name'] for c in trace['captures'] for s in c['scopes']}
            self.assertTrue({'prune.selection','prune.compact_rows','prune.restore_output','decoder.qkv_projections'}<=labels)
            self.assertTrue(all(not p['pruning'].get('banks') for p in bench['policies']))

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_delayed_pruning_matches_explicit_dense_kv_removal(self):
        from experiments.audio_drift.compact import PrefixCache
        from experiments.audio_drift.decode import forward_logits
        from experiments.audio_drift.hooks import blocks_of
        self.prepared = Prepared(self.prepared.prompt, torch.randn(1,4,16).to(torch.bfloat16),1,2,31)
        self.llm.float()
        inputs = [torch.tensor([[1, 2, 3, 31, 31, 31, 31]]) for _ in range(4)]
        inputs[2][0, 3] = 5; inputs[3][0, 4] = 6
        for dtype in [torch.float32, torch.bfloat16]:
            self.llm.to(dtype)
            for scope, selection, interval in [('audio','attention',1), ('prompt','attention',1),
                    ('prefix','attention',1), ('acoustic','attention',1), ('semantic','attention',1), ('audio','random',2)]:
                policy = dict(start=1, fraction=.5, layer_start=0, queries=3,
                              seed=1729, scope=scope, selection=selection)
                schedule = [dict(group='prefix', refresh_every=interval)]
                with PrefixCache(self.llm, self.prepared, schedule, pruning=policy) as cache:
                    actual = [forward_logits(self.llm, self.prepared, x) for x in inputs]
                    counts = cache.accounting()
                    dropped = cache.pruning.dropped[0]
                    self.assertEqual(cache.pruning.dropped[1], dropped)
                    size=len(self.prepared.group_positions('both' if scope=='audio' else scope))
                    self.assertEqual(len(dropped),int(size*.5))
                    self.assertTrue(set(dropped) <= set(self.prepared.group_positions('both' if scope=='audio' else scope)))
                    self.assertEqual([e['layer_kv_columns'] for e in cache.events], [[11,11]]+[[11-len(dropped)]*2]*3)
                    self.assertEqual(counts['pruning']['activated_layers'], 2)
                    self.assertEqual(len(counts['pruning']['selections']), 1)
                    self.assertTrue(all(p >= self.prepared.answer_start for p in counts['pruning']['selections'][0]['probe_positions']))
                # Independent reference: compute every row, remove exactly the same
                # absolute KV columns in SDPA, leave upstream RoPE and MLP intact.
                step = [-1]; originals = []
                kept = torch.tensor([p for p in range(11) if p not in dropped])
                try:
                    for block in blocks_of(self.llm):
                        original = block._scaled_dot_product_attention
                        existed = '_scaled_dot_product_attention' in block.__dict__
                        originals.append((block, original, existed))
                        def masked(q,k,v,_original=original,**kw):
                            if step[0]>=1:
                                k,v=k.index_select(-2,kept),v.index_select(-2,kept)
                            return _original(q,k,v,**kw)
                        block._scaled_dot_product_attention=masked
                    with PrefixCache(self.llm,self.prepared,schedule):
                        reference=[]
                        for i,x in enumerate(inputs):
                            step[0]=i
                            reference.append(forward_logits(self.llm,self.prepared,x))
                finally:
                    for block,original,existed in originals:
                        if existed: block._scaled_dot_product_attention=original
                        else: delattr(block,'_scaled_dot_product_attention')
                for a,b in zip(actual,reference):
                    torch.testing.assert_close(a,b,rtol=0.03 if dtype==torch.bfloat16 else 1e-5,
                                               atol=0.03 if dtype==torch.bfloat16 else 1e-5)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_pruning_benchmark_replay_and_report_retain_scope_pairs(self):
        from types import SimpleNamespace
        from experiments.audio_drift.run import parser, write_json
        from experiments.audio_drift.benchmark import benchmark
        from experiments.audio_drift.replay import benchmark_replay
        from experiments.audio_drift.pruning_report import analyze
        self.prepared = Prepared(self.prepared.prompt, torch.randn(1,4,16).to(torch.bfloat16),1,2,31)
        self.llm.float()
        args=parser().parse_args(['--manifest','unused','--out','unused', '--interventions','freeze',
            '--freeze-groups','prefix','--refresh-intervals','1','8', '--condition-names','freeze_prefix_r1','freeze_prefix_r8',
            '--late-prune','--prune-starts','1','--prune-fractions','0.5','--prune-layer-start','0',
            '--length','4','--block-length','2','--steps','4','--benchmark-repeats','1','--replay-repeats','1'])
        final,states,_=decode(self.llm,self.prepared,4,2,4)
        tokenizer=SimpleNamespace(eos_token_id=30,decode=lambda ids,**kw:'#### 1')
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda,'synchronize'), \
                patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_allocated',return_value=0):
            root=Path(directory);work=root/'samples/00000';work.mkdir(parents=True)
            bench=benchmark(self.llm,self.prepared,final,args,tokenizer,{'id':'tiny','metric':'numeric','target':'1'},work)
            replay=benchmark_replay(self.llm,self.prepared,states,bench['baseline_forwards'],args,work,sample_id='tiny')
            self.assertEqual(len(bench['policies']),12)
            self.assertEqual(len(replay['policies']),12)
            for policy in bench['policies']:
                if policy.get('pruning'):
                    selection=policy['accounting'][0]['pruning']['selections'][0]
                    scope=policy['pruning']['scope']
                    size=len(self.prepared.group_positions('both' if scope=='audio' else scope))
                    self.assertEqual(selection['drop_budget'],int(size*.5))
                    self.assertEqual(selection['anchor_progress']['revealed_fraction'],.25)
                    self.assertEqual(sum(selection['dropped_by_group'].values()),selection['drop_budget'])
            write_json(work/'result.json',{'id':'tiny','score':{'correct':True},'benchmark':bench})
            rows=analyze(root)
            self.assertEqual(len(rows),10)
            self.assertEqual({r['scope'] for r in rows},{'acoustic','semantic','audio','prompt','prefix'})
            self.assertTrue(all(r['activated'] and r['replay_activated'] for r in rows))
            self.assertTrue((root/'analysis/pruning_report.md').exists())

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_pruning_waits_for_anchor_respects_layer_and_zero_budget(self):
        from experiments.audio_drift.compact import PrefixCache
        from experiments.audio_drift.decode import forward_logits
        self.prepared = Prepared(self.prepared.prompt, torch.randn(1,4,16).to(torch.bfloat16),1,2,31)
        self.llm.float()
        x=torch.tensor([[1,2,3,31,31,31,31]])
        schedule=[dict(group='prefix',refresh_every=1)]
        expected=forward_logits(self.llm,self.prepared,x)
        for fraction,start in [(.1,1),(.5,5)]:
            p=dict(start=start,fraction=fraction,layer_start=1,queries=2,seed=1729,scope='audio',selection='attention')
            with PrefixCache(self.llm,self.prepared,schedule,pruning=p) as cache:
                for _ in range(3):
                    torch.testing.assert_close(forward_logits(self.llm,self.prepared,x),expected,rtol=0,atol=0)
                self.assertTrue(all(e['layer_kv_columns']==[11,11] for e in cache.events))
        p.update(fraction=.5,start=1)
        with PrefixCache(self.llm,self.prepared,schedule,pruning=p) as cache:
            for _ in range(3): forward_logits(self.llm,self.prepared,x)
            self.assertEqual([e['layer_computed_rows'] for e in cache.events],[[11,11],[11,11],[11,9]])
            self.assertEqual([e['layer_kv_columns'] for e in cache.events],[[11,11],[11,9],[11,9]])
        with self.assertRaisesRegex(ValueError,'layer'):
            PrefixCache(self.llm,self.prepared,schedule,pruning={**p,'layer_start':2})

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_attention_diagnostics_preserve_logits_and_skip_changed_queries(self):
        from experiments.audio_drift.attention_dynamics import AttentionDynamics
        from experiments.audio_drift.decode import forward_logits
        self.llm.float()
        inputs = [torch.tensor([[1, 2, 3, 31, 31, 31, 31]]) for _ in range(3)]
        inputs[-1][0, 3] = 5
        expected = [forward_logits(self.llm, self.prepared, x) for x in inputs]
        with tempfile.TemporaryDirectory() as directory:
            with AttentionDynamics(self.llm, self.prepared, [0, 1], [5, 6], Path(directory), True) as observer:
                for step, x in enumerate(inputs):
                    observer.begin({'step': step}, x)
                    actual = forward_logits(self.llm, self.prepared, x)
                    torch.testing.assert_close(actual, expected[step], rtol=0, atol=0)
                    observer.end()
            rows = [json.loads(s) for s in (Path(directory)/'attention_dynamics.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            same = [r for r in rows if r['step'] == 1]
            self.assertTrue(all(r['hidden_cosine_distance'] < 1e-6 and r['attention_l1_change'] == 0 for r in same))
            changed = [r for r in rows if r['step'] == 2]
            self.assertTrue(all(r['unchanged_query_count'] == 1 for r in changed))
            self.assertTrue(all(r['count'] == (3 if r['group'] == 'prompt' else 1) for r in changed))
            self.assertTrue((Path(directory)/'attention_token_metrics/0002.npz').exists())
            with np.load(Path(directory)/'attention_token_metrics/0000.npz') as first:
                self.assertEqual(int(first['attention_schema_version']), 2)
                self.assertEqual(int(first['sequence_length']), 9)
                self.assertIn('l00_acoustic_attention_all_probe_mean', first)
                self.assertIn('l00_acoustic_attention_all_probe_peak', first)
            self.assertTrue(all(not b._forward_hooks for b in observer.blocks))
            torch.testing.assert_close(forward_logits(self.llm, self.prepared, inputs[0]), expected[0], rtol=0, atol=0)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_refresh_profile_row_budget_against_actual_execution(self):
        from types import SimpleNamespace
        import time
        from experiments.audio_drift.refresh_profile import profile_refresh
        self.llm.float()
        _, states, _ = decode(self.llm, self.prepared, 4, 2, 4)
        args = SimpleNamespace(interventions=['freeze'], freeze_groups=['prefix'], refresh_intervals=[1, 2],
            mixed_refresh=[], triple_refresh=['0:2:0'], condition_names=['freeze_prefix_r1', 'freeze_prefix_r2',
            'freeze_triple_a0_p2_s0'], benchmark_conditions=[], length=4,
            refresh_profile_steps=4, refresh_profile_repeats=2, refresh_trace=False)
        class Event:
            def __init__(self, **kw): pass
            def record(self): self.t = time.perf_counter()
            def synchronize(self): pass
            def elapsed_time(self, other): return (other.t - self.t)*1000
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, 'synchronize'), \
                patch.object(torch.cuda, 'Event', Event):
            report = profile_refresh(self.llm, self.prepared, states, 4, args, Path(directory), 'tiny')
            self.assertEqual(len(report['conditions']), 4)
            for item in report['conditions']:
                self.assertEqual(len(item['records']), 8)
                self.assertGreater(item['flop_estimate']['total_estimated_flops'], 0)
            permanent = report['conditions'][-1]
            self.assertEqual(permanent['row_budget']['refreshes_by_group']['acoustic'], [1, 1])
            self.assertIn('prompt', permanent['phases'])

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_attention_replay_report_directory_and_zip_match(self):
        from types import SimpleNamespace
        import zipfile
        from experiments.audio_drift.attention_dynamics import collect_dynamics
        from experiments.audio_drift.diagnostic_report import analyze
        from experiments.audio_drift.attention_recurrence import analyze as recurrence
        self.llm.float()
        final, states, events = decode(self.llm, self.prepared, 4, 2, 4)
        args = SimpleNamespace(diagnostic_queries=2, diagnostic_layers='all', diagnostic_tokens=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'run'; work = root/'samples/00000'; work.mkdir(parents=True)
            (root/'config.json').write_text('{}')
            collect_dynamics(self.llm, self.prepared, states, events, final[0, 3:].tolist(), 30, args, work)
            result = {'id':'tiny', 'cohort':'test', 'forwards':4, 'answer_ids':final[0, 3:].tolist(),
                      'labels':self.prepared.labels(4), 'prompt_tokens':3, 'audio_acoustic_tokens':1,
                      'audio_semantic_tokens':1, 'baseline_content_revealed_step':3}
            (work/'result.json').write_text(json.dumps(result))
            direct = analyze(root, Path(directory)/'direct', plots=False)
            direct_risk = recurrence(root, Path(directory)/'direct', plots=False)
            self.assertEqual(direct_risk['samples'], 1)
            self.assertEqual(direct_risk['unsupported_samples'], [])
            self.assertGreater(direct['attention_rows'], 0)
            self.assertTrue((Path(directory)/'direct/diagnostic_correlations.csv').exists())
            archive = Path(directory)/'run.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for p in root.rglob('*'):
                    if p.is_file(): z.write(p, 'run/'+str(p.relative_to(root)))
            zipped = analyze(archive, Path(directory)/'zipped', plots=False)
            self.assertEqual(direct, zipped)
            self.assertEqual(direct_risk, recurrence(archive, Path(directory)/'zipped', plots=False))

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_profiler_scopes_restore_compact_cache_methods(self):
        from experiments.audio_drift.compact import PrefixCache
        from experiments.audio_drift.profile import decoder_scopes
        from experiments.audio_drift.decode import forward_logits
        self.llm.float()
        _, states, _ = decode(self.llm, self.prepared, 4, 2, 4)
        policy = conditions(['freeze'], triple_refresh=['0:2:0'])[-1]
        expected = []
        with PrefixCache(self.llm, self.prepared, policy['schedule']):
            for x in states: expected.append(forward_logits(self.llm, self.prepared, x).clone())
        with PrefixCache(self.llm, self.prepared, policy['schedule']):
            for x, reference in zip(states, expected):
                with decoder_scopes(self.llm): actual = forward_logits(self.llm, self.prepared, x)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_pairwise_collection_preserves_baseline_and_writes_selected_layers(self):
        from experiments.audio_drift.pairwise import PairwiseRecorder
        baseline, _, _ = decode(self.llm, self.prepared, 4, 2, 4)
        with tempfile.TemporaryDirectory() as directory:
            labels = self.prepared.labels(4)
            pairwise = PairwiseRecorder(directory, labels, [0, 2], [0, 1], "cpu")
            recorder = DriftRecorder(directory, labels, 31, pairwise=pairwise)
            try:
                with AttentionProbe(self.llm, self.prepared.audio_positions, recorder):
                    measured, _, _ = decode(self.llm, self.prepared, 4, 2, 4, recorder, [0, 1])
            finally:
                recorder.close()
            torch.testing.assert_close(baseline, measured, rtol=0, atol=0)
            data = json.loads((Path(directory) / "pairwise_cosine.json").read_text())
            self.assertEqual(len(data["records"]), 6)
            self.assertEqual(data["steps"], [0, 2])
            self.assertTrue(data["complete"])

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_compact_matches_dense_kv_reuse_and_skips_projection_rows(self):
        from contextlib import ExitStack
        from experiments.audio_drift.compact import PrefixCache
        from experiments.audio_drift.decode import forward_logits
        self.llm.float()
        inputs = [torch.tensor([[1, 2, 3, 31, 31, 31, 31]]),
                  torch.tensor([[1, 2, 3, 4, 31, 31, 31]]),
                  torch.tensor([[1, 2, 3, 4, 5, 31, 31]]),
                  torch.tensor([[1, 2, 3, 4, 5, 6, 31]]),
                  torch.tensor([[1, 2, 3, 4, 5, 6, 7]])]
        self.assertEqual(self.prepared.group_positions("prompt"), [0, 3, 4])
        self.assertEqual(self.prepared.group_positions("audio_matched"), [1, 2])
        schedules = [[{"group": g, "refresh_every": r}] for g in
                     ["prompt", "both", "prefix", "audio_matched"] for r in [0, 1, 4]]
        schedules.append([{"group": "acoustic", "refresh_every": 0},
                          {"group": "semantic", "refresh_every": 2}])
        schedules.append([{"group": "acoustic", "refresh_every": 8},
                          {"group": "prompt", "refresh_every": 4},
                          {"group": "semantic", "refresh_every": 4}])
        for policy in schedules:
            with self.subTest(policy=policy):
                with AttentionProbe(self.llm, list(range(self.prepared.answer_start)), mode="capture") as capture:
                    first = forward_logits(self.llm, self.prepared, inputs[0])
                dense = [first]
                with ExitStack() as stack:
                    for spec in policy:
                        positions = self.prepared.group_positions(spec["group"])
                        indices = torch.tensor(positions)
                        bank = {layer: tuple(t.index_select(-2, indices) for t in pair)
                                for layer, pair in capture.frozen.items()}
                        stack.enter_context(AttentionProbe(self.llm, positions, mode="freeze", frozen=bank,
                                                           refresh_every=spec["refresh_every"]))
                    dense.extend(forward_logits(self.llm, self.prepared, x) for x in inputs[1:])
                rows = []
                handle = self.llm.model.transformer.blocks[0].ff_proj.register_forward_pre_hook(
                    lambda module, args: rows.append(args[0].shape[1]))
                try:
                    with PrefixCache(self.llm, self.prepared, policy) as cache:
                        compact = [forward_logits(self.llm, self.prepared, x) for x in inputs]
                    self.assertEqual(rows, [e["computed_rows"] for e in cache.events])
                    self.assertEqual(rows[0], 9)
                    self.assertTrue(all(n >= 4 for n in rows))  # every answer row stays active
                    if any(s["refresh_every"] != 1 for s in policy):
                        self.assertLess(min(rows), 9)
                    for a, b in zip(dense, compact):
                        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
                finally:
                    handle.remove()
                # Exception cleanup must restore the original model too.
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    with PrefixCache(self.llm, self.prepared, policy):
                        raise RuntimeError("stop")
                torch.testing.assert_close(first, forward_logits(self.llm, self.prepared, inputs[0]), rtol=0, atol=0)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_depth_cache_matches_dense_state_and_kv_freezing(self):
        from contextlib import contextmanager
        from experiments.audio_drift.compact import PrefixCache
        from experiments.audio_drift.decode import forward_logits
        self.llm.float()
        inputs=[torch.tensor([[1,2,3,4+i%5,10+(i//3)%4,31,31]]) for i in range(17)]
        cases=[
            [{'group':'prefix','refresh_every':1,'refresh_by_layer_group':[4,1]}],
            [{'group':'acoustic','refresh_every':2,'refresh_by_layer_group':[8,2]},
             {'group':'prompt','refresh_every':2,'refresh_by_layer_group':[8,2]},
             {'group':'semantic','refresh_every':1,'refresh_by_layer_group':[4,1]}],
            [{'group':'acoustic','refresh_every':0,'refresh_by_layer_group':[0,0]},
             {'group':'prompt','refresh_every':2,'refresh_by_layer_group':[8,2]},
             {'group':'semantic','refresh_every':1,'refresh_by_layer_group':[4,1]}]]

        @contextmanager
        def dense_reference(policy):
            # Independent oracle computes ALL rows, substitutes inactive K/V,
            # then freezes inactive block outputs. Unlike KV-only substitution,
            # this correctly models a deep refresh fed by a cached shallow state.
            originals=[];banks={};clock=[-1]
            for layer,block in enumerate(self.llm.model.transformer.blocks):
                original=block.forward;originals.append((block,original))
                def forward(x,*a,_l=layer,_b=block,_orig=original,**kw):
                    if _l==0:clock[0]+=1
                    inactive=[]
                    for item in policy:
                        interval=item['refresh_by_layer_group'][_l]
                        if clock[0] and (not interval or clock[0]%interval):
                            inactive.extend(self.prepared.group_positions(item['group']))
                    active=torch.tensor([i for i in range(x.shape[1]) if i not in inactive])
                    attention=_b._scaled_dot_product_attention
                    def substitute(q,k,v,**kwargs):
                        if _l not in banks:
                            banks[_l]=[k.clone(),v.clone(),None]
                        else:
                            banks[_l][0].index_copy_(-2,active,k.index_select(-2,active))
                            banks[_l][1].index_copy_(-2,active,v.index_select(-2,active))
                        return attention(q,banks[_l][0],banks[_l][1],**kwargs)
                    _b._scaled_dot_product_attention=substitute
                    try:output=_orig(x,*a,**kw)
                    finally:_b._scaled_dot_product_attention=attention
                    if banks[_l][2] is None:banks[_l][2]=output[0].clone()
                    else:banks[_l][2].index_copy_(1,active,output[0].index_select(1,active))
                    return (banks[_l][2],*output[1:])
                block.forward=forward
            try:yield
            finally:
                for block,original in originals:block.forward=original

        for policy in cases:
            with self.subTest(policy=policy):
                with dense_reference(policy):
                    expected=[forward_logits(self.llm,self.prepared,x) for x in inputs]
                with PrefixCache(self.llm,self.prepared,policy) as cache:
                    actual=[forward_logits(self.llm,self.prepared,x) for x in inputs]
                for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
                counts=cache.accounting()
                self.assertEqual(counts['forwards'],17)
                self.assertEqual(counts['computed_token_layers'],sum(sum(e['layer_computed_rows']) for e in cache.events))
                self.assertEqual(counts['dense_token_layers_at_same_forward_count'],17*9*2)
                self.assertTrue(all(all(n>=4 for n in e['layer_computed_rows']) for e in cache.events))
                self.assertLess(counts['computed_rows_by_layer'][0],counts['computed_rows_by_layer'][1])
                # Every shallow refresh for a token group also refreshes deeper layers.
                for e in cache.events:
                    for shallow,deep in zip(e['layer_refresh'][0],e['layer_refresh'][1]):
                        if shallow:self.assertTrue(deep)
                with self.assertRaisesRegex(RuntimeError,'cleanup'):
                    with PrefixCache(self.llm,self.prepared,policy):raise RuntimeError('cleanup')
                baseline=forward_logits(self.llm,self.prepared,inputs[0])
                torch.testing.assert_close(baseline,actual[0],rtol=0,atol=0)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_eos_stopping_waits_for_preceding_masks(self):
        from experiments.audio_drift.decode import forward_logits
        # Force confidence order: EOS at position 1, then position 0, then tail.
        logits = torch.full((1, 4, 32), -10.)
        for pos, token, strength in [(0, 5, 8.), (1, 30, 12.), (2, 6, 4.), (3, 7, 2.)]:
            logits[0, pos, token] = strength
        with patch("experiments.audio_drift.decode.forward_logits", return_value=logits):
            fixed, _, _ = decode(self.llm, self.prepared, 4, 4, 4, save_states=False)
            early, _, events = decode(self.llm, self.prepared, 4, 4, 4, save_states=False, eos_id=30)
        self.assertEqual(len(events), 2)
        self.assertEqual(early[0, 3:].tolist(), [5, 30, 31, 31])
        self.assertEqual(fixed[0, 3:5].tolist(), early[0, 3:5].tolist())

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_cache_benchmark_writes_paired_quality_and_actual_timings(self):
        from experiments.audio_drift.benchmark import benchmark
        from types import SimpleNamespace
        self.llm.float()
        baseline, _, _ = decode(self.llm, self.prepared, 4, 2, 4)
        args = SimpleNamespace(interventions=["freeze"], freeze_groups=["both", "prompt", "audio_matched"],
                               refresh_intervals=[1, 4], mixed_refresh=[],
                               condition_names=["freeze_both_r1", "freeze_prompt_r4", "freeze_audio_matched_r4"],
                               benchmark_conditions=[], benchmark_stopping="eos", benchmark_repeats=2,
                               length=4, block_length=2, steps=4)
        tokenizer = SimpleNamespace(eos_token_id=30, decode=lambda ids, **kwargs: str(ids))
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, "synchronize"), \
                patch.object(torch.cuda, "reset_peak_memory_stats"), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=0):
            data = benchmark(self.llm, self.prepared, baseline, args, tokenizer,
                             {"metric": "none"}, Path(directory))
            self.assertEqual(len(data["policies"]), 3)
            self.assertEqual(len(data["baseline_seconds"]), 2)
            for p in data["policies"]:
                self.assertGreater(p["decode_speedup"], 0)
                self.assertEqual(len(p["seconds"]), 2)
                self.assertEqual(len(p["accounting"]), 2)
                self.assertEqual(p["accounting"][0]["forwards"], data["baseline_forwards"])
            identity = data["policies"][0]
            self.assertTrue(identity["same_visible_tokens"])
            self.assertEqual(identity["accounting"][0]["skipped_token_update_fraction"], 0)
            self.assertTrue((Path(directory) / "cache_benchmark.json").is_file())

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_fixed_input_replay_keeps_inputs_and_forward_counts_equal(self):
        from experiments.audio_drift.replay import benchmark_replay
        from experiments.audio_drift.decode import forward_logits
        from types import SimpleNamespace
        self.llm.float()
        _, states, _ = decode(self.llm, self.prepared, 4, 2, 4)
        original = [s.clone() for s in states]
        args = SimpleNamespace(interventions=["freeze"], freeze_groups=["prefix"],
                               refresh_intervals=[1, 8], mixed_refresh=[], triple_refresh=["0:8:8"],
                               condition_names=["freeze_prefix_r1", "freeze_prefix_r8", "freeze_triple_a0_p8_s8"],
                               benchmark_conditions=[], replay_steps=3, replay_repeats=2)
        seen = []
        def observe(llm, prepared, x):
            seen.append(x.clone())
            return forward_logits(llm, prepared, x)
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, "synchronize"), \
                patch.object(torch.cuda, "reset_peak_memory_stats"), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                patch("experiments.audio_drift.replay.forward_logits", side_effect=observe):
            report = benchmark_replay(self.llm, self.prepared, states, 2, args, Path(directory))
            self.assertEqual(report['forwards'], 2)  # cap at baseline EOS, not requested three
            self.assertEqual(len(report['policies']), 3)
            self.assertEqual(report['step_indices'], [0, 1])
            self.assertTrue(report['policies'][0]['identity_last_step_top1_match'])
            self.assertEqual(len(seen), (3 + 3*3)*2)
            for i, x in enumerate(seen):
                torch.testing.assert_close(x, original[i % 2], rtol=0, atol=0)
            for p in report['policies']:
                self.assertGreater(p['forward_replay_speedup'], 0)
                self.assertTrue(all(c['forwards'] == 2 for c in p['accounting']))
            self.assertTrue((Path(directory)/'fixed_input_replay.json').is_file())
        for x,y in zip(states,original):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        with self.assertRaises(ValueError):
            benchmark_replay(self.llm, self.prepared, [], 0, args, Path('/tmp'))

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_depth_generation_replay_and_cohort_report_together(self):
        from types import SimpleNamespace
        from experiments.audio_drift.benchmark import benchmark
        from experiments.audio_drift.replay import benchmark_replay
        from experiments.audio_drift.reasoning_report import summarize, compare_depth, main
        self.llm.float();baseline,states,_=decode(self.llm,self.prepared,4,2,4)
        args=SimpleNamespace(interventions=['freeze'],freeze_groups=['prefix'],refresh_intervals=[1,2],
            mixed_refresh=[],triple_refresh=[],condition_names=['freeze_prefix_r1','freeze_prefix_r2'],
            depth_multipliers=[2,1],benchmark_conditions=[],benchmark_stopping='eos',benchmark_repeats=2,
            replay_steps=3,replay_repeats=2,length=4,block_length=2,steps=4)
        tokenizer=SimpleNamespace(eos_token_id=30,decode=lambda ids,**kw:'#### 18')
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda,'synchronize'), \
             patch.object(torch.cuda,'reset_peak_memory_stats'), \
             patch.object(torch.cuda,'max_memory_allocated',return_value=0):
            directory=Path(directory)
            bench=benchmark(self.llm,self.prepared,baseline,args,tokenizer,{'metric':'numeric','target':'18'},directory)
            replay=benchmark_replay(self.llm,self.prepared,states,bench['baseline_forwards'],args,directory)
            self.assertEqual(len(bench['policies']),3)
            results=[]
            for i,cohort in enumerate(['development','fresh']):
                result={'id':str(i),'cohort':cohort,'target':'18','score':score('#### 18','18','numeric'),
                        'answer_has_no_eos':False,'benchmark':bench,'fixed_input_replay':replay}
                results.append(result)
                folder=directory/'samples'/f'{i:05}';folder.mkdir(parents=True)
                (folder/'result.json').write_text(json.dumps(result))
            rows,summary=summarize(results)
            depth='freeze_prefix_r2_depth2x1'
            self.assertEqual(summary[depth]['cohorts']['fresh']['samples'],1)
            self.assertEqual(summary[depth]['cohorts']['development']['samples'],1)
            comparisons=compare_depth(rows)
            self.assertEqual(comparisons['all'][depth]['samples'],2)
            self.assertEqual(comparisons['fresh'][depth]['depth_correct'],1)
            self.assertGreater(comparisons['fresh'][depth]['replay_depth_speedup_over_flat'],0)
            with patch('sys.argv',['reasoning_report','--run',str(directory)]):main()
            self.assertIn('Flat versus depth-aware: fresh',(directory/'analysis/reasoning_report.md').read_text())

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_no_drift_runner_never_constructs_measurement_hooks(self):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from experiments.audio_drift import run
        self.llm.float()
        tokenizer=SimpleNamespace(eos_token_id=30,all_special_ids=[30,31],decode=lambda ids,**kw:str(ids))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            directory=Path(directory);audio=directory/'fake.wav';audio.write_bytes(b'fixture')
            manifest=directory/'input.jsonl'
            manifest.write_text(json.dumps({'id':'fixture','audio':'fake.wav','task':'fixture','prompt':'test',
                                            'cohort':'fresh'})+'\n')
            out=directory/'run'
            argv=['audio_drift','--manifest',str(manifest),'--out',str(out),
                  '--diffa-root',os.environ['DIFFA_ROOT'],'--length','4','--block-length','2','--steps','4',
                  '--no-drift','--verify-samples','0']
            stack.enter_context(patch('sys.argv',argv))
            for name in ['synchronize','reset_peak_memory_stats','manual_seed_all']:
                stack.enter_context(patch.object(torch.cuda,name))
            for name,value in [('is_bf16_supported',True),('max_memory_allocated',0),
                               ('get_device_name','CPU fixture'),('get_device_properties',SimpleNamespace(total_memory=0))]:
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            # Pass the runner's hardware gate once; CUDA stays unavailable to CPU TorchDynamo.
            stack.enter_context(patch.object(torch.cuda,'is_available',side_effect=[True]+[False]*10000))
            stack.enter_context(patch.object(run.importlib.metadata,'version',return_value='fixture'))
            stack.enter_context(patch.object(run,'resolve_revision',return_value='fixture'))
            stack.enter_context(patch.object(run,'load_model',return_value=(SimpleNamespace(llm_model=self.llm),tokenizer,None,{})))
            stack.enter_context(patch.object(run,'load_audio',return_value=(np.zeros(16),.001)))
            stack.enter_context(patch.object(run,'prepare',return_value=self.prepared))
            stack.enter_context(patch.object(run,'DriftRecorder',side_effect=AssertionError('drift disabled')))
            stack.enter_context(patch.object(run,'AttentionProbe',side_effect=AssertionError('drift disabled')))
            run.main()
            result=json.loads((out/'samples/00000/result.json').read_text())
            self.assertFalse(result['drift_recorded'])
            self.assertEqual(result['cohort'],'fresh')
            self.assertFalse((out/'samples/00000/drift.jsonl').exists())

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_profiler_scopes_preserve_outputs_and_restore_on_exception(self):
        from experiments.audio_drift.profile import decoder_scopes
        from experiments.audio_drift.decode import forward_logits
        x = torch.tensor([[1, 2, 3, 31, 31, 31, 31]])
        baseline = forward_logits(self.llm, self.prepared, x)
        block = self.llm.model.transformer.blocks[0]
        original_keys = set(block.q_proj.__dict__)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with decoder_scopes(self.llm):
                actual = forward_logits(self.llm, self.prepared, x)
        torch.testing.assert_close(baseline, actual, rtol=0, atol=0)
        names = {e.key for e in prof.key_averages()}
        self.assertTrue({"decoder.qkv_projections", "decoder.feed_forward_projections",
                         "decoder.attention_kernel", "decoder.vocabulary_projection"} <= names)
        # Verify the timing fields exist on the installed torch API too.
        self.assertTrue(all(hasattr(e, "device_time_total") and hasattr(e, "self_device_time_total")
                            for e in prof.key_averages()))
        with self.assertRaisesRegex(RuntimeError, "scope failure"):
            with decoder_scopes(self.llm): raise RuntimeError("scope failure")
        self.assertEqual(set(block.q_proj.__dict__), original_keys)
        torch.testing.assert_close(baseline, forward_logits(self.llm, self.prepared, x), rtol=0, atol=0)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_baseline_profile_exports_trace_and_checks_clean_decode(self):
        from experiments.audio_drift.profile import profile_baseline
        from types import SimpleNamespace
        baseline, states, _ = decode(self.llm, self.prepared, 4, 2, 4)
        args = SimpleNamespace(length=4, block_length=2, steps=4)
        real_profile = torch.profiler.profile
        def cpu_profile(**kwargs):
            kwargs["activities"] = [torch.profiler.ProfilerActivity.CPU]
            return real_profile(**kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                patch("experiments.audio_drift.model.prepare", return_value=self.prepared), \
                patch.object(torch.cuda, "synchronize"), \
                patch.object(torch.cuda, "reset_peak_memory_stats"), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                patch.object(torch.profiler, "profile", side_effect=cpu_profile):
            root = Path(directory)
            profile_baseline(SimpleNamespace(llm_model=self.llm), SimpleNamespace(eos_token_id=30),
                             None, None, "Transcribe", self.prepared, baseline, states, args, root)
            data = json.loads((root / "baseline_profile.json").read_text())
            self.assertEqual(len(data["decoding_seconds"]), 2)
            self.assertEqual(data["sampled_forward_steps"], [0, 1, 3])
            self.assertTrue(data["scopes"])
            self.assertTrue((root / "baseline_trace.json").is_file())
            self.assertTrue((root / "baseline_operators.txt").is_file())

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_instrumented_decoder_matches_official_and_observes_all_layers(self):
        from src.origin_generate import generate
        from types import SimpleNamespace
        official = generate(SimpleNamespace(llm_model=self.llm, device=torch.device("cpu")),
                            self.prepared.prompt, torch.tensor([1]), self.prepared.speech,
                            steps=4, gen_length=4, block_length=2, mask_id=31)
        with tempfile.TemporaryDirectory() as directory:
            recorder = DriftRecorder(directory, self.prepared.labels(4), 31)
            with AttentionProbe(self.llm, self.prepared.audio_positions, recorder):
                actual, _, _ = decode(self.llm, self.prepared, 4, 2, 4, recorder, [0, 1])
            recorder.close()
            self.assertTrue(torch.equal(official, actual))
            rows = [json.loads(line) for line in (Path(directory) / "drift.jsonl").read_text().splitlines()]
            self.assertEqual({r["tensor"] for r in rows}, {"hidden", "key", "value"})
            self.assertEqual({r["layer"] for r in rows}, {0, 1})
            first_layer_keys = [r for r in rows if r["layer"] == 0 and r["tensor"] == "key"
                                and r["group"] in {"audio", "prompt"}]
            self.assertTrue(all(r["metrics"]["relative_l2"]["max"] == 0 for r in first_layer_keys))

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_capture_freeze_remove_and_exception_cleanup(self):
        from experiments.audio_drift.decode import forward_logits
        x = torch.tensor([[1, 2, 3, 31, 31]])
        baseline = forward_logits(self.llm, self.prepared, x)
        with AttentionProbe(self.llm, self.prepared.audio_positions, mode="capture") as capture:
            observed = forward_logits(self.llm, self.prepared, x)
        torch.testing.assert_close(baseline, observed, rtol=0, atol=0)
        with AttentionProbe(self.llm, self.prepared.audio_positions, mode="freeze", frozen=capture.frozen):
            frozen = forward_logits(self.llm, self.prepared, x)
        torch.testing.assert_close(baseline, frozen, rtol=0, atol=0)
        with AttentionProbe(self.llm, self.prepared.audio_positions, mode="remove"):
            removed = forward_logits(self.llm, self.prepared, x)
        self.assertFalse(torch.equal(baseline, removed))
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with AttentionProbe(self.llm, self.prepared.audio_positions, mode="remove"):
                raise RuntimeError("test failure")
        torch.testing.assert_close(baseline, forward_logits(self.llm, self.prepared, x), rtol=0, atol=0)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_all_continuations_start_from_the_paired_state(self):
        from experiments.audio_drift.run import interventions
        from types import SimpleNamespace
        baseline, states, _ = decode(self.llm, self.prepared, 4, 2, 4)
        args = SimpleNamespace(interventions=["freeze", "remove", "text_only"],
                               branch_fractions=[0., .5], length=4, block_length=2, steps=4)
        tokenizer = SimpleNamespace(eos_token_id=30,
                                    decode=lambda ids, **kwargs: " ".join(map(str, ids)))
        with patch("torch.cuda.synchronize"):
            persisted = []
            results = interventions(self.llm, self.prepared, baseline, states,
                                     args, tokenizer, {"metric": "none"}, on_result=persisted.append)
            args.cache_device = "model"
            device_results = interventions(self.llm, self.prepared, baseline, states,
                                            args, tokenizer, {"metric": "none"})
        self.assertEqual(len(persisted), len(results))
        for cpu, model in zip(results, device_results):
            self.assertEqual(cpu["answer_ids"], model["answer_ids"])
            self.assertEqual(cpu["immediate"], model["immediate"])
            self.assertEqual(cpu["cache_events"], model["cache_events"])
        self.assertEqual(len(results), 6)
        for result in results:
            start = result["start_step"]
            self.assertEqual(result["immediate"]["count"], 4 - start)
            for position, token in enumerate(states[start][0, 3:].tolist()):
                if token != 31:
                    self.assertEqual(result["answer_ids"][position], token)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_group_only_freeze_and_refresh_do_not_mutate_anchor_bank(self):
        from experiments.audio_drift.decode import forward_logits
        anchor = torch.tensor([[1, 2, 3, 31, 31]])
        changed = torch.tensor([[1, 2, 3, 4, 31]])
        positions = self.prepared.group_positions("acoustic")
        self.assertEqual(positions, [1])
        self.assertEqual(self.prepared.group_positions("semantic"), [2])
        with AttentionProbe(self.llm, positions, mode="capture") as capture:
            forward_logits(self.llm, self.prepared, anchor)
        original = {layer: tuple(t.clone() for t in pair) for layer, pair in capture.frozen.items()}
        expected = forward_logits(self.llm, self.prepared, changed)
        with AttentionProbe(self.llm, positions, mode="freeze", frozen=capture.frozen,
                            refresh_every=2, anchor_step=7) as reuse:
            forward_logits(self.llm, self.prepared, changed)  # step 8: reuse
            refreshed = forward_logits(self.llm, self.prepared, changed)  # step 9: refresh
        torch.testing.assert_close(expected, refreshed, rtol=0, atol=0)
        self.assertEqual(reuse.cache_events, [{"step": 8, "refreshed": False, "cache_age": 1},
                                            {"step": 9, "refreshed": True, "cache_age": 0}])
        for layer, pair in original.items():
            for before, after in zip(pair, capture.frozen[layer]):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_only_selected_group_kvs_are_replaced(self):
        from experiments.audio_drift.decode import forward_logits
        from types import MethodType
        block = self.llm.model.transformer.blocks[1]
        original = block._scaled_dot_product_attention
        received = []
        def spy(_block, q, k, v, **kwargs):
            received.append((k.clone(), v.clone()))
            return original(q, k, v, **kwargs)
        block._scaled_dot_product_attention = MethodType(spy, block)
        anchor = torch.tensor([[1, 2, 3, 31, 31]])
        changed = torch.tensor([[1, 2, 3, 4, 31]])
        try:
            with AttentionProbe(self.llm, [1], mode="capture") as capture:
                forward_logits(self.llm, self.prepared, anchor)
            anchor_kv = received[-1]
            forward_logits(self.llm, self.prepared, changed)
            fresh_kv = received[-1]
            with AttentionProbe(self.llm, [1], mode="freeze", frozen=capture.frozen):
                forward_logits(self.llm, self.prepared, changed)
            used_kv = received[-1]
            for old, fresh, used in zip(anchor_kv, fresh_kv, used_kv):
                torch.testing.assert_close(used[:, :, 1], old[:, :, 1], rtol=0, atol=0)
                torch.testing.assert_close(used[:, :, 2], fresh[:, :, 2], rtol=0, atol=0)
            self.assertFalse(torch.equal(anchor_kv[0][:, :, 1], fresh_kv[0][:, :, 1]))
        finally:
            block._scaled_dot_product_attention = original

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_group_matrix_including_mixed_schedules(self):
        from experiments.audio_drift.run import interventions
        from types import SimpleNamespace
        baseline, states, _ = decode(self.llm, self.prepared, 4, 2, 4)
        args = SimpleNamespace(interventions=["freeze"], branch_fractions=[0.], length=4,
                               block_length=2, steps=4, freeze_groups=["acoustic", "semantic", "both"],
                               refresh_intervals=[1, 2], mixed_refresh=["2:1", "1:2"])
        tokenizer = SimpleNamespace(eos_token_id=30, all_special_ids=[30, 31],
                                    decode=lambda ids, **kwargs: " ".join(map(str, ids)))
        with patch("torch.cuda.synchronize"):
            results = interventions(self.llm, self.prepared, baseline, states,
                                     args, tokenizer, {"metric": "none"})
            args.cache_device = "model"
            device_results = interventions(self.llm, self.prepared, baseline, states,
                                            args, tokenizer, {"metric": "none"})
        for cpu, model in zip(results, device_results):
            self.assertEqual(cpu["answer_ids"], model["answer_ids"])
            self.assertEqual(cpu["immediate"], model["immediate"])
            self.assertEqual(cpu["cache_events"], model["cache_events"])
        self.assertEqual(len(results), 11)
        for result in results:
            if result["condition"].endswith("_r1"):
                self.assertTrue(result["same_final_tokens"])
                self.assertEqual(result["immediate"]["kl_mean"], 0)
            for group, events in result["cache_events"].items():
                interval = next(s["refresh_every"] for s in result["schedule"] if s["group"] == group)
                self.assertEqual(len(events), 3)
                self.assertEqual([e["step"] for e in events], [1, 2, 3])
                self.assertEqual([e["refreshed"] for e in events],
                                 [bool(interval and s % interval == 0) for s in [1, 2, 3]])
                self.assertLessEqual(result["immediate_content"]["count"], result["immediate"]["count"])

    @torch.no_grad() if HAS_TORCH else lambda f: f
    def test_real_connectors_and_prompt_alignment(self):
        from src.configuration_DIFFA import DIFFAConfig
        from src.modeling_DIFFA import DIFFAModel
        from transformers import WhisperFeatureExtractor
        from experiments.audio_drift.model import prepare
        from types import SimpleNamespace
        config = DIFFAConfig(whisper_config=dict(d_model=16, encoder_layers=12,
                            encoder_attention_heads=2, encoder_ffn_dim=32,
                            num_mel_bins=4, max_source_positions=50),
                             llm_config=self.llm.config.to_dict())
        config.whisper_model_id = "openai/whisper-small"
        config.prompt_size = 64  # upstream's validity mask explicitly assumes 64
        def tokenize(text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]),
                                   attention_mask=torch.ones(1, 3, dtype=torch.long))
        # Special methods are resolved on the class, not instance attributes.
        class Tokenizer:
            eos_token = "EOS"
            def __call__(self, text, **kwargs):
                return tokenize(text, **kwargs)
        tokenizer = Tokenizer()
        model = DIFFAModel(config, tokenizer=tokenizer).eval()
        model.llm_model = self.llm
        model.llm_model.config.max_sequence_length = 256
        extractor = WhisperFeatureExtractor(feature_size=4, sampling_rate=16000,
                                            hop_length=160, chunk_length=1, n_fft=400)
        prepared = prepare(model, tokenizer, extractor, np.zeros(3200, dtype=np.float32), "Transcribe", 4)
        self.assertEqual(prepared.insertion, 2)
        self.assertEqual(prepared.acoustic_count, 64)
        self.assertEqual(prepared.audio_count, 66)
        self.assertEqual(len(prepared.labels(4)), 73)


if __name__ == "__main__":
    unittest.main()
