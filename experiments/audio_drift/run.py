"""Collect DIFFA audio drift on a local JSONL manifest; run inference on Vast."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
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
from .protocol import anchors_for, conditions, work_estimate
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


def intervention_specs(args):
    specs = conditions(args.interventions, getattr(args, "freeze_groups", ["both"]),
                       getattr(args, "refresh_intervals", []), getattr(args, "mixed_refresh", []))
    names = getattr(args, "condition_names", [])
    if names:
        unknown = set(names) - {s["condition"] for s in specs}
        if unknown:
            raise ValueError(f"Requested conditions not in generated matrix: {sorted(unknown)}")
        specs = [s for s in specs if s["condition"] in names]
    return specs


@torch.no_grad()
def interventions(llm, prepared, baseline, states, args, tokenizer, row, on_result=None):
    specifications = intervention_specs(args)
    if not specifications:
        return []
    results = []
    anchors = anchors_for(len(states), args.branch_fractions)
    total = len(anchors) * len(specifications)
    baseline_ids = baseline[0, prepared.prompt.shape[1]:].tolist()
    stop = baseline_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in baseline_ids else len(baseline_ids)
    special = set(getattr(tokenizer, "all_special_ids", [tokenizer.eos_token_id]))
    lexical = torch.tensor([[i < stop and token not in special for i, token in enumerate(baseline_ids)]])
    for anchor in anchors:
        start = anchor + 1
        anchor_x = states[anchor].to(llm.device)
        cache_device = getattr(args, "cache_device", "cpu")
        captured_positions = list(range(prepared.answer_start))
        with AttentionProbe(llm, captured_positions, mode="capture", cache_device=cache_device) as capture:
            anchor_logits = forward_logits(llm, prepared, anchor_x)
        del anchor_logits, anchor_x
        start_x = states[start].to(llm.device)
        reference = forward_logits(llm, prepared, start_x).cpu()
        remaining = start_x[:, prepared.prompt.shape[1]:].cpu() == prepared.mask_id
        content_remaining = remaining & lexical
        content_masks = int(content_remaining.sum())
        for specification in specifications:
            name, mode = specification["condition"], specification["mode"]
            print(f"  branch {len(results) + 1}/{total}: {name}; anchor={anchor}; "
                  f"{content_masks} baseline content masks remain", flush=True)
            branch_prepared = prepared
            probes = []
            torch.cuda.synchronize()
            began = time.perf_counter()
            with ExitStack() as stack:
                if mode == "text_only":
                    branch_prepared = Prepared(prepared.prompt, prepared.speech[:, :0],
                                               prepared.insertion, 0, prepared.mask_id)
                elif mode == "remove":
                    stack.enter_context(AttentionProbe(llm, prepared.audio_positions, mode="remove"))
                else:
                    for schedule in specification["schedule"]:
                        group = schedule["group"]
                        positions = prepared.group_positions(group)
                        if not positions:
                            raise ValueError(f"No tokens in requested group {group}")
                        # The capture bank contains the whole fixed prefix. Select
                        # only this group's columns in the SAME position order.
                        indices = torch.tensor(positions)
                        bank = {layer: tuple(t.index_select(-2, indices.to(t.device)) for t in pair)
                                for layer, pair in capture.frozen.items()}
                        probe = stack.enter_context(AttentionProbe(llm, positions, mode="freeze",
                                      frozen=bank, refresh_every=schedule["refresh_every"], anchor_step=anchor,
                                      cache_device=cache_device))
                        probes.append((group, probe))
                    del bank
                first = forward_logits(llm, branch_prepared, start_x)
                first_cpu = first.cpu()
                immediate = distribution_change(reference, first_cpu, remaining)
                immediate_content = distribution_change(reference, first_cpu, content_remaining)
                del first_cpu
                final, _, branch_events = decode(llm, branch_prepared, args.length, args.block_length,
                                      args.steps, start_x=start_x, start_step=start,
                                      first_logits=first, save_states=False)
            torch.cuda.synchronize()
            text = display_text(tokenizer, final, prepared.prompt.shape[1])
            result = {**specification, "anchor_step": anchor,
                      "anchor_step_fraction": anchor / max(1, len(states) - 1),
                      "start_step": start, "remaining_fraction": float(remaining.float().mean()),
                      "baseline_content_masks_remaining": content_masks,
                      "baseline_answer_already_revealed": content_masks == 0,
                      "immediate": immediate, "immediate_content": immediate_content, "text": text,
                      "answer_ids": final[0, prepared.prompt.shape[1]:].tolist(),
                      "same_final_tokens": torch.equal(final, baseline),
                      "score": score(text, row.get("target"), row.get("metric", "none")),
                      "cache_events": {group: probe.cache_events for group, probe in probes},
                      "cached_group_positions": {group: prepared.group_positions(group) for group, _ in probes},
                      "commit_events": [{"step": e["step"], "positions": e["committed_positions"]}
                                        for e in branch_events],
                      "diagnostic_elapsed_seconds": time.perf_counter() - began}
            # Refresh-every-forward is an identity control: it must not change
            # decoding. This guards schedule/mapping correctness on the GPU too.
            if mode == "freeze" and all(s["refresh_every"] == 1 for s in specification["schedule"]):
                if not result["same_final_tokens"]:
                    raise RuntimeError(f"Refresh-every-forward control differs from baseline: {name}")
            results.append(result)
            if on_result is not None:
                on_result(result)
            print(f"    done in {result['diagnostic_elapsed_seconds']:.1f}s: "
                  f"same_tokens={result['same_final_tokens']}; score={result['score']}", flush=True)
            del first, final, probes
        del reference, capture, start_x
    return results


def summarize_pairs(entries):
    scored = [e for e in entries if e["score"] is not None]
    result = {"samples": len(entries), "scored": len(scored)}
    correct = [e["score"]["correct"] for e in scored if "correct" in e["score"]]
    if correct:
        result["accuracy"] = sum(correct) / len(correct)
    wer = [e for e in scored if e["score"]["metric"] == "wer"]
    if wer:
        denominator = max(1, sum(e["score"]["reference_words"] for e in wer))
        result["corpus_wer"] = sum(e["score"]["edits"] for e in wer) / denominator
        result["paired_baseline_corpus_wer"] = sum(e["baseline_score"]["edits"] for e in wer) / denominator
        result["corpus_wer_delta"] = result["corpus_wer"] - result["paired_baseline_corpus_wer"]
    if entries and all("same_final_tokens" in e for e in entries):
        result["identical_tokens_fraction"] = sum(e["same_final_tokens"] for e in entries) / len(entries)
    return result


def aggregate(output):
    tasks = defaultdict(lambda: {"samples": 0, "conditions": defaultdict(list)})
    benchmarks = defaultdict(list)
    for path in sorted((output / "samples").glob("*/result.json")):
        if path.parent.name.startswith("."):
            continue
        result = json.loads(path.read_text())
        task = tasks[result["task"]]
        task["samples"] += 1
        bench = result.get("benchmark")
        if bench and "baseline_median_seconds" in bench:
            for policy in bench["policies"]:
                benchmarks[result["task"], policy["condition"]].append((bench, policy, result["score"]))
        task["conditions"]["baseline"].append({"score": result["score"], "baseline_score": result["score"]})
        for branch in result["interventions"]:
            key = f"{branch.get('condition', branch['mode'])}@{branch['anchor_step_fraction']:.3f}"
            task["conditions"][key].append({**branch, "baseline_score": result["score"]})
    summary = {}
    for name, task in tasks.items():
        conditions_summary = {}
        for name2, entries in task["conditions"].items():
            counts = summarize_pairs(entries)
            if name2 != "baseline":
                unfinished = [e for e in entries if not e["baseline_answer_already_revealed"]]
                counts["with_content_remaining"] = summarize_pairs(unfinished)
                counts["already_revealed_samples"] = len(entries) - len(unfinished)
            conditions_summary[name2] = counts
        summary[name] = {"samples": task["samples"], "conditions": conditions_summary}
    bench_summary = {}
    for (task, condition), pairs in benchmarks.items():
        quality = summarize_pairs([{"score": p["score"], "baseline_score": baseline} for _, p, baseline in pairs])
        bench_summary.setdefault(task, {})[condition] = {**quality,
            "decode_speedup_ratio_of_sums": sum(b["baseline_median_seconds"] for b, _, _ in pairs) /
                                             sum(p["median_seconds"] for _, p, _ in pairs),
            "same_visible_tokens_fraction": sum(p["same_visible_tokens"] for _, p, _ in pairs) / len(pairs),
            "stopping": sorted({b["stopping"] for b, _, _ in pairs})}
    write_json(output / "summary.json", {"tasks": summary, "cache_benchmarks": bench_summary,
               "timing_note": "Diagnostic runtimes are instrumented. cache_benchmarks, when present, report "
                              "separate warmed timings with real row skipping and an explicit stopping rule.",
               "quality_note": "Compare the with_content_remaining subset; completed answers can hide damage."})


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
    p.add_argument("--cache-device", choices=["cpu", "model"], default="model",
                   help="Intervention KV storage: model keeps it on the model's GPU; independent of drift references")
    p.add_argument("--token-metrics", action="store_true",
                   help="Also save compressed scalar metrics per position (larger outputs)")
    p.add_argument("--snapshot-steps", type=int, nargs="*", default=[])
    p.add_argument("--snapshot-layers", type=int, nargs="*", default=[])
    p.add_argument("--interventions", nargs="*", choices=["freeze", "remove", "text_only"], default=[])
    p.add_argument("--freeze-groups", nargs="+",
                   choices=["acoustic", "semantic", "both", "prompt", "prefix", "audio_matched"], default=["both"])
    p.add_argument("--refresh-intervals", type=int, nargs="*", default=[],
                   help="Also test periodic refresh; 1 is an identity control, 0/permanent is automatic")
    p.add_argument("--mixed-refresh", nargs="*", default=[], metavar="ACOUSTIC:SEMANTIC",
                   help="Independent intervals for the two groups, e.g. 8:2 2:8; 0 means no refresh")
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--condition-names", nargs="*", default=[],
                   help="Run only these named conditions from the generated matrix")
    p.add_argument("--profile-samples", type=int, default=0,
                   help="Profile this many initial samples separately, without drift hooks")
    p.add_argument("--benchmark-samples", type=int, default=0)
    p.add_argument("--benchmark-repeats", type=int, default=2)
    p.add_argument("--benchmark-conditions", nargs="*", default=[])
    p.add_argument("--benchmark-stopping", choices=["eos", "fixed"], default="eos")
    p.add_argument("--branch-fractions", type=float, nargs="+", default=[0., .25, .5, .75])
    p.add_argument("--verify-samples", type=int, default=1,
                   help="Compare instrumented decoding with the official unmodified decoder")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--cpu-threads", type=int, default=4,
                   help="Limit CPU tensor math threads to avoid oversubscribing small diagnostic operations")
    return p


def main():
    args = parser().parse_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    schedule(args.length, args.block_length, args.steps)
    specifications = intervention_specs(args)
    if args.benchmark_samples and set(args.benchmark_conditions) - {s["condition"] for s in specifications if s["mode"] == "freeze"}:
        raise ValueError("Benchmark conditions must select generated freeze conditions")
    estimate = work_estimate(args.steps, args.branch_fractions, specifications)
    print("Work per example:", json.dumps(estimate), flush=True)
    if args.progress_every < 0:
        raise ValueError("progress-every must be nonnegative")
    if min(args.verify_samples, args.profile_samples, args.benchmark_samples) < 0 or args.benchmark_repeats < 1 or not all(0 <= f < 1 for f in args.branch_fractions):
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
              "python": platform.python_version(), "protocol": "audio-drift-v2",
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
        sample_began = time.perf_counter()
        print(f"[{index + 1}/{len(rows)}] {row['id']} ({row['task']})", flush=True)
        audio, duration = load_audio(row["audio"])
        torch.cuda.synchronize()
        preparation_began = time.perf_counter()
        prepared = prepare(model, tokenizer, extractor, audio, row["prompt"], args.length)
        torch.cuda.synchronize()
        preparation_seconds = time.perf_counter() - preparation_began
        recorder = DriftRecorder(work, prepared.labels(args.length), prepared.mask_id,
                                 args.reference_device, args.token_metrics,
                                 args.snapshot_steps, args.snapshot_layers)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = time.perf_counter()
        try:
            with AttentionProbe(llm, prepared.audio_positions, recorder, layers):
                final, states, events = decode(llm, prepared, args.length, args.block_length,
                                               args.steps, recorder, layers, progress_every=args.progress_every)
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
        text = display_text(tokenizer, final, prepared.prompt.shape[1])
        answer_ids = final[0, prepared.prompt.shape[1]:].tolist()
        first_eos = answer_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in answer_ids else None
        visible = set()
        content_positions = {i for i, token in enumerate(answer_ids)
                             if (first_eos is None or i < first_eos)
                             and token not in set(tokenizer.all_special_ids)}
        content_revealed_step = None
        for event in events:
            visible.update(event["committed_positions"])
            if content_positions <= visible:
                content_revealed_step = event["step"]
                break
        baseline_score = score(text, row.get("target"), row.get("metric", "none"))
        print(f"  baseline: official_match={verified}; score={baseline_score}; text={text!r}", flush=True)
        write_json(work / "steps.json", events)
        result = {"id": row["id"], "task": row["task"], "prompt": row["prompt"],
                  "target": row.get("target"), "audio_seconds": duration,
                  "audio_preparation_seconds": preparation_seconds,
                  "duration_bin": row.get("duration_bin"), "speaker_id": row.get("speaker_id"),
                  "audio_semantic_tokens": prepared.audio_count - prepared.acoustic_count,
                  "audio_acoustic_tokens": prepared.acoustic_count,
                  "prompt_tokens": prepared.prompt.shape[1], "labels": prepared.labels(args.length),
                  "text": text, "answer_ids": answer_ids,
                  "first_eos_answer_position": first_eos,
                  "answer_has_no_eos": first_eos is None,
                  "baseline_content_revealed_step": content_revealed_step,
                  "score": baseline_score,
                  "verified_against_upstream": verified, "forwards": len(events),
                  "instrumented_seconds": elapsed, "peak_allocated_bytes": peak_memory,
                  "interventions": []}
        write_json(work / "baseline.json", result)
        if index < args.profile_samples:
            from .profile import profile_baseline
            profile_baseline(model, tokenizer, extractor, audio, row["prompt"], prepared,
                             final, states, args, work)
        # Persist each completed branch even if the sample is interrupted later.
        with (work / "interventions.jsonl").open("w") as branch_file:
            def save_branch(branch):
                branch_file.write(json.dumps(branch, allow_nan=False) + "\n")
                branch_file.flush()
            result["interventions"] = interventions(llm, prepared, final, states, args, tokenizer, row,
                                                     on_result=save_branch)
        if index < args.benchmark_samples:
            from .benchmark import benchmark
            result["benchmark"] = benchmark(llm, prepared, final, args, tokenizer, row, work,
                                             result["interventions"], preparation_seconds)
        result["sample_wall_seconds"] = time.perf_counter() - sample_began
        print(f"  sample complete in {result['sample_wall_seconds'] / 60:.1f} minutes", flush=True)
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
