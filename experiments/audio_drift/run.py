"""Collect DIFFA audio drift on a local JSONL manifest; run inference on Vast."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import torch

from .checkout import COMMIT, DEFAULT_ROOT, verify_checkout
from .decode import Prepared, decode, distribution_change, forward_logits, schedule
from .hooks import AttentionProbe, blocks_of
from .inputs import load_audio, read_manifest, score
from .metrics import DriftRecorder
from .model import (DIFFA_REPO, DIFFA_REVISION, LLM_REPO, LLM_REVISION,
                    load_model, prepare, resolve_revision)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def display_text(tokenizer, x, prompt_length):
    ids = x[0, prompt_length:].tolist()
    # Scoring stops at the first EOS even though decoding fills all slots.
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


@torch.no_grad()
def interventions(llm, prepared, baseline, states, args, tokenizer, row):
    if not args.interventions:
        return []
    from .hooks import AttentionProbe
    results = []
    anchors = sorted({round(f * (len(states) - 1)) for f in args.branch_fractions})
    for anchor in anchors:
        start = anchor + 1
        if start >= len(states):
            continue
        anchor_x = states[anchor].to(llm.device)
        # Capture the exact baseline audio memory after normalization and RoPE.
        with AttentionProbe(llm, prepared.audio_positions, mode="capture") as capture:
            anchor_logits = forward_logits(llm, prepared, anchor_x)
        del anchor_logits, anchor_x
        start_x = states[start].to(llm.device)
        reference = forward_logits(llm, prepared, start_x).cpu()
        remaining = start_x[:, prepared.prompt.shape[1]:].cpu() == prepared.mask_id
        baseline_ids = baseline[0, prepared.prompt.shape[1]:].tolist()
        stop = baseline_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in baseline_ids else len(baseline_ids)
        # Posthoc diagnostic only: don't count already-fixed answers as evidence
        # that audio can be removed while meaningful generation remains.
        content_masks = int(remaining[0, :stop].sum())
        for mode in args.interventions:
            branch_prepared = prepared
            if mode == "text_only":
                # Cold recomputation from IDs. Audio is absent; later RoPE
                # positions shift, so this is a text-sufficiency diagnostic,
                # not a clean estimate of audio attention's marginal effect.
                branch_prepared = Prepared(prepared.prompt, prepared.speech[:, :0],
                                           prepared.insertion, 0, prepared.mask_id)
                context = AttentionProbe(llm, [], mode="observe")
            else:
                context = AttentionProbe(llm, prepared.audio_positions, mode=mode,
                                         frozen=capture.frozen)
            torch.cuda.synchronize()
            began = time.perf_counter()
            with context:
                first = forward_logits(llm, branch_prepared, start_x)
                immediate = distribution_change(reference, first.cpu(), remaining)
                final, _, _ = decode(llm, branch_prepared, args.length, args.block_length,
                                      args.steps, start_x=start_x, start_step=start,
                                      first_logits=first, save_states=False)
            torch.cuda.synchronize()
            text = display_text(tokenizer, final, prepared.prompt.shape[1])
            results.append({"mode": mode, "anchor_step": anchor,
                            "anchor_step_fraction": anchor / max(1, len(states) - 1),
                            "start_step": start, "remaining_fraction": float(remaining.float().mean()),
                            "baseline_content_masks_remaining": content_masks,
                            "baseline_answer_already_revealed": content_masks == 0,
                            "immediate": immediate, "text": text,
                            "answer_ids": final[0, prepared.prompt.shape[1]:].tolist(),
                            "same_final_tokens": torch.equal(final, baseline),
                            "score": score(text, row.get("target"), row.get("metric", "none")),
                            "diagnostic_elapsed_seconds": time.perf_counter() - began})
            del first, final
        del reference, capture, start_x
    return results


def aggregate(output):
    tasks = defaultdict(lambda: {"samples": 0, "conditions": defaultdict(list)})
    for path in sorted((output / "samples").glob("*/result.json")):
        result = json.loads(path.read_text())
        task = tasks[result["task"]]
        task["samples"] += 1
        task["conditions"]["baseline"].append(result["score"])
        for branch in result["interventions"]:
            key = f"{branch['mode']}@{branch['anchor_step_fraction']:.3f}"
            task["conditions"][key].append(branch["score"])
    summary = {}
    for name, task in tasks.items():
        conditions = {}
        for name2, scores in task["conditions"].items():
            usable = [s for s in scores if s is not None]
            counts = {"samples": len(scores), "scored": len(usable)}
            correct = [s["correct"] for s in usable if "correct" in s]
            wer = [s for s in usable if s["metric"] == "wer"]
            if correct:
                counts["accuracy"] = sum(correct) / len(correct)
            if wer:
                counts["corpus_wer"] = sum(s["edits"] for s in wer) / max(1, sum(s["reference_words"] for s in wer))
            conditions[name2] = counts
        summary[name] = {"samples": task["samples"], "conditions": conditions}
    write_json(output / "summary.json", {"tasks": summary,
               "timing_note": "Instrumented diagnostic runtime; no optimized-cache speedup claim."})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diffa-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--diffa-repo", default=DIFFA_REPO)
    p.add_argument("--diffa-revision", default=DIFFA_REVISION)
    p.add_argument("--llm-repo", default=LLM_REPO)
    p.add_argument("--llm-revision", default=LLM_REVISION)
    p.add_argument("--whisper-repo", default="openai/whisper-large-v3")
    p.add_argument("--whisper-revision", default="main")
    p.add_argument("--lora-alpha", type=int, default=16,
                   help="Matches upstream DIFFA-2 inference; saved in load report")
    p.add_argument("--samples", type=int)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--layers", default="all", help="all or comma-separated 0-based layers")
    p.add_argument("--reference-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--token-metrics", action="store_true",
                   help="Also save compressed scalar metrics per position (larger outputs)")
    p.add_argument("--snapshot-steps", type=int, nargs="*", default=[])
    p.add_argument("--snapshot-layers", type=int, nargs="*", default=[])
    p.add_argument("--interventions", nargs="*", choices=["freeze", "remove", "text_only"], default=[])
    p.add_argument("--branch-fractions", type=float, nargs="+", default=[0., .25, .5, .75])
    p.add_argument("--verify-samples", type=int, default=1,
                   help="Compare instrumented decoding with the official unmodified decoder")
    p.add_argument("--seed", type=int, default=1729)
    return p


def main():
    args = parser().parse_args()
    schedule(args.length, args.block_length, args.steps)
    if args.verify_samples < 0 or not all(0 <= f < 1 for f in args.branch_fractions):
        raise ValueError("Invalid verification count or branch fractions")
    if bool(args.snapshot_steps) != bool(args.snapshot_layers):
        raise ValueError("Specify both snapshot steps and layers")
    verify_checkout(args.diffa_root)
    rows = read_manifest(args.manifest, args.samples)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required; run this on Vast")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    revisions = {"diffa": resolve_revision(args.diffa_repo, args.diffa_revision),
                 "llm": resolve_revision(args.llm_repo, args.llm_revision),
                 "whisper": resolve_revision(args.whisper_repo, args.whisper_revision)}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    fingerprint = hashlib.sha256()
    for path in sorted([*Path(__file__).parent.glob("*.py"), Path(__file__).with_name("requirements.txt")]):
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    config = {"settings": settings, "resolved_revisions": revisions, "source_sha256": fingerprint.hexdigest(),
              "upstream_commit": COMMIT, "inputs": rows, "torch": torch.__version__,
              "python": platform.python_version(), "protocol": "audio-drift-v1",
              "packages": {name: importlib.metadata.version(name) for name in
                           ("transformers", "peft", "numpy", "datasets", "huggingface-hub", "torchaudio")}}
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Run config/source/inputs changed; use a new output directory")
    else:
        if any(args.out.iterdir()):
            raise ValueError("Output directory is not empty and has no run config")
        write_json(config_path, config)
    model, tokenizer, extractor, report = load_model(
        args.diffa_root, args.diffa_repo, revisions["diffa"], args.llm_repo, revisions["llm"],
        args.whisper_repo, revisions["whisper"], lora_alpha=args.lora_alpha)
    write_json(args.out / "load_report.json", report)
    llm = model.llm_model
    layers = list(range(len(blocks_of(llm)))) if args.layers == "all" else sorted(set(map(int, args.layers.split(","))))
    if not layers or min(layers) < 0 or max(layers) >= len(blocks_of(llm)):
        raise ValueError("Invalid layer selection")
    if not set(args.snapshot_layers) <= set(layers) or any(s < 0 or s >= args.steps for s in args.snapshot_steps):
        raise ValueError("Snapshots must select observed layers and valid steps")
    (args.out / "samples").mkdir(exist_ok=True)
    write_json(args.out / "device.json", {"name": torch.cuda.get_device_name(),
               "vram_bytes": torch.cuda.get_device_properties(0).total_memory})
    from src.origin_generate import generate as upstream_generate
    from types import SimpleNamespace
    for index, row in enumerate(rows):
        destination = args.out / "samples" / f"{index:05d}"
        if (destination / "result.json").exists():
            print(f"Skip completed {row['id']}", flush=True)
            continue
        work = args.out / "samples" / f".{index:05d}.incomplete"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        print(f"[{index + 1}/{len(rows)}] {row['id']} ({row['task']})", flush=True)
        audio, duration = load_audio(row["audio"])
        prepared = prepare(model, tokenizer, extractor, audio, row["prompt"], args.length)
        recorder = DriftRecorder(work, prepared.labels(args.length), prepared.mask_id,
                                 args.reference_device, args.token_metrics,
                                 args.snapshot_steps, args.snapshot_layers)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = time.perf_counter()
        try:
            with AttentionProbe(llm, prepared.audio_positions, recorder, layers):
                final, states, events = decode(llm, prepared, args.length, args.block_length,
                                               args.steps, recorder, layers)
        finally:
            recorder.close()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        peak_memory = torch.cuda.max_memory_allocated()
        verified = None
        if index < args.verify_samples:
            official = upstream_generate(SimpleNamespace(llm_model=llm, device=llm.device),
                        prepared.prompt, prepared.prompt.new_tensor([prepared.insertion]), prepared.speech,
                        steps=args.steps, gen_length=args.length, block_length=args.block_length,
                        temperature=0., cfg_scale=0., mask_id=prepared.mask_id)
            verified = torch.equal(final, official)
            if not verified:
                raise RuntimeError("Instrumented output differs from the official decoder; stopping")
            del official
        branches = interventions(llm, prepared, final, states, args, tokenizer, row)
        text = display_text(tokenizer, final, prepared.prompt.shape[1])
        write_json(work / "steps.json", events)
        result = {"id": row["id"], "task": row["task"], "prompt": row["prompt"],
                  "target": row.get("target"), "audio_seconds": duration,
                  "audio_semantic_tokens": prepared.audio_count - prepared.acoustic_count,
                  "audio_acoustic_tokens": prepared.acoustic_count,
                  "prompt_tokens": prepared.prompt.shape[1], "labels": prepared.labels(args.length),
                  "text": text, "answer_ids": final[0, prepared.prompt.shape[1]:].tolist(),
                  "score": score(text, row.get("target"), row.get("metric", "none")),
                  "verified_against_upstream": verified, "forwards": len(events),
                  "instrumented_seconds": elapsed, "peak_allocated_bytes": peak_memory,
                  "interventions": branches}
        write_json(work / "result.json", result)
        if destination.exists():
            raise ValueError(f"Unexpected existing incomplete destination: {destination}")
        work.rename(destination)
        aggregate(args.out)
        del final, states, prepared, recorder
    aggregate(args.out)
    print(f"Complete: {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
