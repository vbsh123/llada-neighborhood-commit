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
