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

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import torch
    from experiments.audio_drift.decode import Prepared, decode, distribution_change, schedule
    from experiments.audio_drift.hooks import AttentionProbe
    from experiments.audio_drift.metrics import DriftRecorder, drift, token_groups


class InputTests(unittest.TestCase):
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
            results = interventions(self.llm, self.prepared, baseline, states,
                                     args, tokenizer, {"metric": "none"})
        self.assertEqual(len(results), 6)
        for result in results:
            start = result["start_step"]
            self.assertEqual(result["immediate"]["count"], 4 - start)
            for position, token in enumerate(states[start][0, 3:].tolist()):
                if token != 31:
                    self.assertEqual(result["answer_ids"][position], token)

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
