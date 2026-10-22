"""Collect DIFFA audio drift on a local JSONL manifest; run inference on Vast."""
import argparse
from collections import defaultdict
from contextlib import ExitStack, nullcontext
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
from .protocol import anchors_for, conditions, work_estimate, with_depth_variants
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
                       getattr(args, "refresh_intervals", []), getattr(args, "mixed_refresh", []),
                       getattr(args, "triple_refresh", []))
    names = getattr(args, "condition_names", [])
    if names:
        unknown = set(names) - {s["condition"] for s in specs}
        if unknown:
            raise ValueError(f"Requested conditions not in generated matrix: {sorted(unknown)}")
        specs = [s for s in specs if s["condition"] in names]
    if getattr(args, 'late_prune', False):
        from .pruning import pruning_conditions
        specs += pruning_conditions(args.prune_starts, args.prune_fractions, args.prune_layer_start,
                                    args.prune_queries, args.seed)
    if getattr(args, 'staged_prune', False):
        from .staged_pruning import staged_conditions
        specs += staged_conditions(args.staged_fractions,args.acoustic_start,args.later_start,args.deep_layer,
                                   args.prune_queries,args.seed)
    if getattr(args, 'crossing_prune', False):
        from .crossing_pruning import crossing_conditions
        specs += crossing_conditions(args.crossing_fractions,args.crossing_layer,args.crossing_start,
                                     args.crossing_interval,args.crossing_patience,args.prune_queries,args.seed)
    factors = getattr(args, 'depth_multipliers', [])
    return with_depth_variants(specs, factors) if factors else specs


@torch.no_grad()
def interventions(llm, prepared, baseline, states, args, tokenizer, row, on_result=None):
    specifications = intervention_specs(args)
    if not specifications or getattr(args, "skip_dense_interventions", False):
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
    numeric = [e["score"] for e in scored if e["score"].get("metric") == "numeric"]
    if numeric:
        result["parse_failures"] = sum(e["parse_failed"] for e in numeric)
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
    p.add_argument("--triple-refresh", nargs="*", default=[], metavar="ACOUSTIC:PROMPT:SEMANTIC",
                   help="Independent fixed-prefix schedules for all three groups")
    p.add_argument('--staged-prune', action='store_true', help='Compact group/layer/time-specific pruning and matched generic controls')
    p.add_argument('--crossing-prune', action='store_true', help='Online audio/prompt attention crossing and event-matched full-prefix control')
    p.add_argument('--crossing-fractions', type=float, nargs='+', default=[.25,.5,.75])
    p.add_argument('--crossing-layer', type=int, default=25)
    p.add_argument('--crossing-start', type=int, default=8)
    p.add_argument('--crossing-interval', type=int, default=8)
    p.add_argument('--crossing-patience', type=int, default=2)
    p.add_argument('--staged-fractions', type=float, nargs='+', default=[.5,.75])
    p.add_argument('--acoustic-start', type=int, default=16)
    p.add_argument('--later-start', type=int, default=32)
    p.add_argument('--deep-layer', type=int, default=16)
    p.add_argument('--pruning-profile-samples', type=int, default=0)
    p.add_argument('--late-prune', action='store_true', help='Add delayed audio and matched-budget pruning controls')
    p.add_argument('--prune-starts', type=int, nargs='+', default=[1, 16, 32], help='Zero-based forward indices; no future-answer dependence')
    p.add_argument('--prune-fractions', type=float, nargs='+', default=[.25], help='Drop quota as a fraction of audio positions')
    p.add_argument('--prune-layer-start', type=int, default=24, help='Prune from this layer through the last layer')
    p.add_argument('--prune-queries', type=int, default=16, help='Fixed online answer-canvas probes, including masks')
    p.add_argument("--skip-dense-interventions", action="store_true",
                   help="Run generated policies only in the compact benchmark, without dense branches")
    p.add_argument("--depth-multipliers", type=int, nargs="*", default=[],
                   help="Add depth-aware variants of selected non-identity policies, e.g. 8 4 2 1")
    p.add_argument("--no-drift", action="store_true",
                   help="Skip drift hooks for a quality/latency experiment after baseline drift collection")
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--condition-names", nargs="*", default=[],
                   help="Run only these named conditions from the generated matrix")
    p.add_argument("--profile-samples", type=int, default=0,
                   help="Profile this many initial samples separately, without drift hooks")
    p.add_argument("--benchmark-samples", type=int, default=0)
    p.add_argument("--benchmark-repeats", type=int, default=2)
    p.add_argument("--benchmark-conditions", nargs="*", default=[])
    p.add_argument("--shuffle-policy-order", action="store_true",
                   help="Shuffle policy timing order reproducibly per sample; identity controls stay first")
    p.add_argument("--replay-samples", type=int, default=0,
                   help="Also time identical baseline input states on this many first benchmark samples")
    p.add_argument("--replay-steps", type=int, default=64,
                   help="Replay this many initial forwards, capped at baseline EOS stopping")
    p.add_argument("--replay-repeats", type=int, default=2)
    p.add_argument("--diagnostic-samples", type=int, default=0,
                   help="Separate untimed attention/drift replay on this many samples")
    p.add_argument("--diagnostic-layers", default="all")
    p.add_argument("--diagnostic-queries", type=int, default=4)
    p.add_argument("--diagnostic-tokens", action="store_true",
                   help="Save scalar per-position drift/attention arrays, never full representations")
    p.add_argument("--refresh-profile-samples", type=int, default=0)
    p.add_argument("--refresh-profile-steps", type=int, default=128)
    p.add_argument("--refresh-profile-repeats", type=int, default=3)
    p.add_argument("--refresh-trace", action="store_true",
                   help="Capture representative operator traces in separate diagnostic passes")
    p.add_argument("--benchmark-stopping", choices=["eos", "fixed"], default="eos")
    p.add_argument("--branch-fractions", type=float, nargs="+", default=[0., .25, .5, .75])
    p.add_argument("--verify-samples", type=int, default=1,
                   help="Compare instrumented decoding with the official unmodified decoder")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--cpu-threads", type=int, default=4,
                   help="Limit CPU tensor math threads to avoid oversubscribing small diagnostic operations")
    p.add_argument("--pairwise-count", type=int, default=0,
                   help="Collect all-pairs cosine on this many baseline hidden-state snapshots; 0 disables")
    p.add_argument("--pairwise-stride", type=int, default=1)
    p.add_argument("--pairwise-layers", default="7,15,23,31")
    p.add_argument("--pairwise-device", choices=["cpu", "cuda"], default="cuda")
    return p


def main():
    args = parser().parse_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    if args.pairwise_count < 0 or args.pairwise_stride < 1:
        raise ValueError("Invalid pairwise count/stride")
    torch.set_num_threads(args.cpu_threads)
    schedule(args.length, args.block_length, args.steps)
    specifications = intervention_specs(args)
    if args.crossing_prune and (args.staged_prune or args.late_prune or not args.skip_dense_interventions
            or args.depth_multipliers or not args.no_drift or args.diagnostic_samples or args.pruning_profile_samples):
        raise ValueError('Crossing pruning requires its own compact-only, no-drift experiment; operator tracing is separate')
    if args.staged_prune and (args.late_prune or not args.skip_dense_interventions or args.depth_multipliers or not args.no_drift or args.diagnostic_samples):
        raise ValueError('Staged pruning requires its own compact-only experiment without cache/depth variants')
    if args.pruning_profile_samples < 0 or args.pruning_profile_samples > args.replay_samples:
        raise ValueError('Pruning profiles require corresponding replay examples')
    if args.late_prune and (not args.skip_dense_interventions or args.depth_multipliers):
        raise ValueError('Delayed pruning requires compact-only policies and no depth variants')
    if args.depth_multipliers and not args.skip_dense_interventions:
        raise ValueError("Depth variants require --skip-dense-interventions; use compact benchmarks/replay")
    if args.no_drift and (args.pairwise_count or args.token_metrics or args.snapshot_steps):
        raise ValueError("--no-drift cannot be combined with drift snapshots, token metrics or pairwise collection")
    if args.benchmark_samples and set(args.benchmark_conditions) - {s["condition"] for s in specifications if s["mode"] == "freeze"}:
        raise ValueError("Benchmark conditions must select generated freeze conditions")
    if (not 0 <= args.replay_samples <= args.benchmark_samples or args.replay_steps < 1
            or args.replay_repeats < 1):
        raise ValueError("Invalid replay settings; replay samples must be benchmark samples")
    if args.replay_samples and args.benchmark_stopping != "eos":
        raise ValueError("Replay requires EOS benchmarks so post-answer steps can be excluded")
    if (args.diagnostic_samples < 0 or
            (args.samples is not None and args.diagnostic_samples > args.samples) or args.diagnostic_queries < 1 or
            not 0 <= args.refresh_profile_samples <= args.replay_samples or
            args.refresh_profile_steps < 1 or args.refresh_profile_repeats < 1):
        raise ValueError("Invalid attention/refresh diagnostic settings")
    if args.refresh_profile_samples and args.depth_multipliers:
        raise ValueError("Per-forward phase profiling currently requires flat refresh schedules")
    if args.skip_dense_interventions and not args.benchmark_samples:
        raise ValueError("--skip-dense-interventions requires --benchmark-samples > 0")
    estimate = work_estimate(args.steps, args.branch_fractions,
                             [] if args.skip_dense_interventions else specifications)
    print("Work per example:", json.dumps(estimate), flush=True)
    if args.diagnostic_samples or args.refresh_profile_samples:
        print(f"Additional diagnostics: attention replay on first {args.diagnostic_samples} samples; "
              f"per-forward refresh profiling on first {args.refresh_profile_samples}, "
              f"up to {args.refresh_profile_steps} baseline pre-EOS states. "
              "These are separate from clean decode/replay timing.", flush=True)
    if args.benchmark_samples:
        names = set(args.benchmark_conditions)
        number = sum(s["mode"] == "freeze" and (not names or s["condition"] in names)
                     for s in specifications)
        print(f"Compact benchmark on first {args.benchmark_samples} samples: {number} policies; "
              f"one warmup + {args.benchmark_repeats} timed decodes per policy; "
              "three clean baseline decodes (warmup and before/after timing).", flush=True)
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
    pairwise_layers = layers if args.pairwise_layers == "all" else sorted(set(map(int, args.pairwise_layers.split(","))))
    if args.pairwise_count and not set(pairwise_layers) <= set(layers):
        raise ValueError("Pairwise layers must also be selected for baseline observation")
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
        pairwise = None
        if args.pairwise_count:
            from .pairwise import PairwiseRecorder
            pair_steps = list(range(0, args.steps, args.pairwise_stride))[:args.pairwise_count]
            pairwise = PairwiseRecorder(work, prepared.labels(args.length), pair_steps, pairwise_layers,
                                        args.pairwise_device)
            estimated = len(pair_steps) * len(pairwise_layers) * prepared.answer_start * llm.config.d_model * 4
            print(f"  pairwise: {len(pair_steps)} snapshots, layers={pairwise_layers}; "
                  f"normalized-vector memory about {estimated / 1e9:.2f} GB on {args.pairwise_device}", flush=True)
        recorder = None if args.no_drift else DriftRecorder(
            work, prepared.labels(args.length), prepared.mask_id, args.reference_device, args.token_metrics,
            args.snapshot_steps, args.snapshot_layers, pairwise=pairwise)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = time.perf_counter()
        try:
            with (AttentionProbe(llm, prepared.audio_positions, recorder, layers)
                  if recorder is not None else nullcontext()):
                final, states, events = decode(llm, prepared, args.length, args.block_length,
                                               args.steps, recorder, layers, progress_every=args.progress_every)
        finally:
            if recorder is not None:
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
                raise RuntimeError("Baseline output differs from the official decoder; stopping")
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
                  "cohort": row.get("cohort", "unspecified"), "drift_recorded": recorder is not None,
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
        if index < args.replay_samples:
            from .replay import benchmark_replay
            result["fixed_input_replay"] = benchmark_replay(
                llm, prepared, states, result["benchmark"]["baseline_forwards"], args, work, sample_id=row["id"])
        if index < args.diagnostic_samples:
            from .attention_dynamics import collect_dynamics
            collect_dynamics(llm, prepared, states, events, answer_ids,
                             tokenizer.eos_token_id, args, work)
        if index < args.pruning_profile_samples:
            from .staged_profile import profile_staged
            profile_staged(llm,prepared,states,result['benchmark']['baseline_forwards'],args,work,row['id'])
        if index < args.refresh_profile_samples:
            from .refresh_profile import profile_refresh
            profile_refresh(llm, prepared, states, result["benchmark"]["baseline_forwards"],
                            args, work, row["id"])
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
