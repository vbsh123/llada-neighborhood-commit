"""Streaming per-position drift; no full attention matrices or activation archive."""
import json
from pathlib import Path

import numpy as np
import torch


def drift(current, reference, eps=1e-8):
    """One FP32 calculation per token, flattening features/heads beforehand.

    Relative L2 uses the reference norm. Two zero vectors have cosine 1;
    one zero vector has cosine 0. Non-finite activations fail the run.
    """
    a, b = current.float(), reference.to(current.device).float()
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Drift requires matching [tokens, features] tensors")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite activation in drift measurement")
    an, bn = a.norm(dim=-1), b.norm(dim=-1)
    cosine = (a * b).sum(-1) / (an * bn).clamp_min(eps)
    cosine = torch.where((an == 0) & (bn == 0), 1.0, cosine).clamp(-1, 1)
    return {
        "relative_l2": (a - b).norm(dim=-1) / bn.clamp_min(eps),
        "cosine_similarity": cosine,
        "cosine_distance": 1 - cosine,
        "zero_reference_norm": (bn == 0).float(),
    }


def token_groups(labels, current_ids, reference_ids, mask_id):
    """Only unchanged identities enter stable masked/decoded controls."""
    current_ids = current_ids.cpu()
    reference_ids = reference_ids.cpu()
    labels = np.asarray(labels)
    answer = torch.from_numpy(labels == "answer")
    same = current_ids == reference_ids
    result = {g: torch.from_numpy(labels == g) for g in
              ("audio_semantic", "audio_acoustic", "prompt")}
    result["audio"] = result["audio_semantic"] | result["audio_acoustic"]
    result["answer_masked"] = answer & same & (current_ids == mask_id)
    result["answer_decoded"] = answer & same & (current_ids != mask_id)
    result["answer_changed"] = answer & ~same
    return result


def summarize(values):
    values = values.float()
    return {"mean": values.mean().item(), "median": torch.quantile(values, .5).item(),
            "p90": torch.quantile(values, .9).item(), "max": values.max().item()}


class DriftRecorder:
    """Reference banks have constant size in the number of decoding steps.

    Keeps previous, first-forward, and block-start values in native dtype on
    CPU by default. Only scalar per-position metrics are persisted to NPZ.
    """
    def __init__(self, output, labels, mask_id, reference_device="cpu",
                 per_token=False, snapshot_steps=(), snapshot_layers=()):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.labels = list(labels)
        self.mask_id = mask_id
        self.reference_device = reference_device
        self.per_token = per_token
        self.snapshot_steps = set(snapshot_steps)
        self.snapshot_layers = set(snapshot_layers)
        self.banks = {name: {} for name in ("previous", "initial", "block_start")}
        self.stream = (self.output / "drift.jsonl").open("w")
        self.seen = set()

    def begin(self, metadata, ids):
        self.metadata = dict(metadata)
        self.ids = ids.detach().flatten().cpu().clone()
        if self.ids.numel() != len(self.labels):
            raise ValueError("Token labels and sequence positions disagree")
        if metadata["step_in_block"] == 0:
            self.banks["block_start"].clear()
        self.arrays = {}
        self.snapshots = {}
        self.seen.clear()

    def observe(self, layer, kind, value):
        if value.ndim != 2 or value.shape[0] != len(self.labels):
            raise ValueError(f"Unexpected {kind} tensor shape: {tuple(value.shape)}")
        key = (layer, kind)
        if key in self.seen:
            raise ValueError(f"Layer observed twice in one forward: {key}")
        self.seen.add(key)
        stored = value.detach().to(self.reference_device).clone()
        now = self.metadata["step"]
        for reference_name, bank in self.banks.items():
            if key not in bank:
                bank[key] = (stored, self.ids, now)
            old, old_ids, old_step = bank[key]
            # CPU references: compute on the copy already made, avoiding a
            # CPU->GPU->CPU round trip for each reference bank.
            measured = stored if self.reference_device == "cpu" else value.detach()
            distances = {k: v.cpu() for k, v in drift(measured, old).items()}
            groups = token_groups(self.labels, self.ids, old_ids, self.mask_id)
            for group, selection in groups.items():
                count = int(selection.sum())
                if not count:
                    continue
                row = {**self.metadata, "layer": layer, "tensor": kind,
                       "reference": reference_name, "reference_step": old_step,
                       "age": now - old_step, "group": group, "count": count,
                       "metrics": {k: summarize(v[selection]) for k, v in distances.items()}}
                self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            if self.per_token:
                prefix = f"l{layer:03d}_{kind}_{reference_name}"
                # IDs/labels permit later regrouping without retaining vectors.
                for metric in ("relative_l2", "cosine_similarity"):
                    self.arrays[f"{prefix}_{metric}"] = distances[metric].numpy()
            if reference_name == "previous":
                bank[key] = (stored, self.ids, now)
        if now in self.snapshot_steps and layer in self.snapshot_layers:
            self.snapshots[f"l{layer:03d}_{kind}"] = stored.cpu().clone()

    def end(self, expected_layers):
        expected = {(layer, kind) for layer in expected_layers for kind in ("hidden", "key", "value")}
        if self.seen != expected:
            raise ValueError(f"Missing/unexpected hooks: {expected ^ self.seen}")
        step = self.metadata["step"]
        if self.per_token:
            path = self.output / "token_metrics"
            path.mkdir(exist_ok=True)
            np.savez_compressed(path / f"{step:04d}.npz", token_ids=self.ids.numpy(),
                                labels=np.asarray(self.labels), **self.arrays)
        if self.snapshots:
            path = self.output / "snapshots"
            path.mkdir(exist_ok=True)
            torch.save({"metadata": self.metadata, "ids": self.ids, "labels": self.labels,
                        "tensors": self.snapshots}, path / f"{step:04d}.pt")
        self.stream.flush()
        self.arrays.clear()
        self.snapshots.clear()

    def close(self):
        self.stream.close()
        for bank in self.banks.values():
            bank.clear()
