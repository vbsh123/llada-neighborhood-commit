"""Exact mean per-token hidden-state cosine across selected baseline steps.

Only normalized fixed-prefix vectors at selected layers/steps are retained in
memory. Disk outputs contain scalar cosine values, never representation vectors.
"""
import json
from pathlib import Path

import numpy as np
import torch


class PairwiseRecorder:
    def __init__(self, output, labels, steps, layers, device="cuda"):
        self.output = Path(output)
        self.steps = sorted(set(steps))
        self.layers = sorted(set(layers))
        if not self.steps or not self.layers or self.steps[0] < 0:
            raise ValueError("Pairwise measurement needs nonnegative steps and observed layers")
        self.index = {step: i for i, step in enumerate(self.steps)}
        self.device = device
        self.groups = {g: torch.tensor([i for i, label in enumerate(labels) if label == g])
                       for g in ("audio_acoustic", "audio_semantic", "prompt")}
        if any(not len(p) for p in self.groups.values()):
            raise ValueError("Pairwise measurement requires acoustic, semantic and prompt tokens")
        self.positions = {}
        self.history, self.zeros, self.matrices, self.counts = {}, {}, {}, {}
        self.stream = (self.output / "pairwise_cosine.jsonl").open("w")
        self.closed = False

    @torch.no_grad()
    def observe(self, layer, kind, value, metadata):
        step = metadata["step"]
        if kind != "hidden" or layer not in self.layers or step not in self.index:
            return
        i = self.index[step]
        for group, positions in self.groups.items():
            key = layer, group
            if self.counts.get(key, 0) != i:
                raise ValueError("Pairwise snapshots must arrive once in increasing selected-step order")
            position_key = group, value.device
            if position_key not in self.positions:
                self.positions[position_key] = positions.to(value.device)
            current = value.detach().index_select(0, self.positions[position_key]).to(self.device, torch.float32)
            if not torch.isfinite(current).all():
                raise ValueError("Non-finite pairwise hidden state")
            norms = current.norm(dim=-1, keepdim=True)
            zero = norms[:, 0] == 0
            normalized = (current / norms.clamp_min(1e-8)).flatten()
            if key not in self.history:
                self.history[key] = normalized.new_empty((len(self.steps), normalized.numel()))
                self.zeros[key] = current.new_empty((len(self.steps), len(positions)))
                self.matrices[key] = np.full((len(self.steps), len(self.steps)), np.nan, dtype=np.float32)
            self.history[key][i].copy_(normalized)
            self.zeros[key][i].copy_(zero)
            # Dot products of flattened per-token unit vectors, divided by token
            # count, equal the mean of individual token cosine similarities.
            cosine = (torch.mv(self.history[key][:i + 1], normalized) +
                      torch.mv(self.zeros[key][:i + 1], zero.float())) / len(positions)
            cosine = cosine.clamp(-1, 1)
            cosine[i] = 1.0
            row = cosine.cpu().numpy()
            self.matrices[key][i, :i + 1] = row
            self.matrices[key][:i + 1, i] = row
            self.counts[key] = i + 1
            self.stream.write(json.dumps({"step": step, "layer": layer, "group": group,
                "count": len(positions), "reference_steps": self.steps[:i + 1],
                "mean_token_cosine_similarity": row.tolist()}, allow_nan=False) + "\n")
            if i + 1 == len(self.steps):
                # Later baseline steps do not require any pairwise vectors.
                del self.history[key], self.zeros[key]

    def flush(self):
        self.stream.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stream.close()
        records = []
        for (layer, group), matrix in sorted(self.matrices.items()):
            records.append({"layer": layer, "group": group, "count": len(self.groups[group]),
                            "cosine_similarity": [[float(v) if np.isfinite(v) else None for v in row]
                                                  for row in matrix]})
        data = {"steps": self.steps, "layers": self.layers,
                "complete": len(self.counts) == len(self.layers) * len(self.groups) and
                            all(n == len(self.steps) for n in self.counts.values()),
                "pooling": "Mean of per-token cosine similarities at matching fixed-prefix positions. "
                           "Normalization and dot products use FP32; both-zero vectors have cosine one. "
                           "Each layer is stored separately. No answer tokens or pooled-embedding cosine. "
                           "This reproduces the figure type; VidLaDA's exact pooling protocol is not established.",
                "records": records}
        path = self.output / "pairwise_cosine.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, allow_nan=False) + "\n")
        temp.replace(path)
        self.history.clear()
        self.zeros.clear()
