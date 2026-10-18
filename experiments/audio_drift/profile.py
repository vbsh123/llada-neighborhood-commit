"""Separate baseline timings and operator traces; never a cache-speedup benchmark."""
from contextlib import contextmanager
import statistics
import time

import torch

from .decode import decode, forward_logits
from .hooks import blocks_of


@contextmanager
def decoder_scopes(llm):
    """Label disjoint projection/attention operations for profiler attribution."""
    originals = []

    def wrap(obj, method, label):
        original = getattr(obj, method)
        # Restore the original instance dictionary, including inherited methods.
        existed = method in obj.__dict__
        originals.append((obj, method, existed, obj.__dict__.get(method)))

        def scoped(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)
        setattr(obj, method, scoped)

    try:
        for block in blocks_of(llm):
            for name in ("q_proj", "k_proj", "v_proj", "att_proj"):
                if hasattr(block, name):
                    wrap(getattr(block, name), "forward", "decoder.qkv_projections")
            for name in ("ff_proj", "up_proj", "ff_out"):
                if hasattr(block, name):
                    wrap(getattr(block, name), "forward", "decoder.feed_forward_projections")
            wrap(block.attn_out, "forward", "decoder.attention_output_projection")
            wrap(block, "_scaled_dot_product_attention", "decoder.attention_kernel")
        if "ff_out" in llm.model.transformer:
            wrap(llm.model.transformer["ff_out"], "forward", "decoder.vocabulary_projection")
        yield
    finally:
        for obj, method, existed, value in reversed(originals):
            if existed:
                setattr(obj, method, value)
            else:
                delattr(obj, method)


@torch.no_grad()
def profile_baseline(model, tokenizer, extractor, audio, prompt, prepared, baseline, states, args, work):
    from .model import prepare
    from .run import write_json

    llm = model.llm_model
    print("  profiling baseline separately (no drift or KV substitution hooks)", flush=True)
    # The instrumented baseline already warmed the exact shapes. Explicitly warm
    # a fresh forward before timing, then use synchronized wall time, no scopes.
    warm = forward_logits(llm, prepared, states[0].to(llm.device))
    del warm
    torch.cuda.synchronize()
    preparation_seconds, decoding_seconds = [], []
    for _ in range(2):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fresh = prepare(model, tokenizer, extractor, audio, prompt, args.length)
        torch.cuda.synchronize()
        preparation_seconds.append(time.perf_counter() - started)
        if not torch.equal(fresh.speech, prepared.speech):
            raise RuntimeError("Audio preparation changed during baseline profiling")
        del fresh
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        torch.cuda.synchronize()
        started = time.perf_counter()
        final, _, _ = decode(llm, prepared, args.length, args.block_length, args.steps, save_states=False)
        torch.cuda.synchronize()
        decoding_seconds.append(time.perf_counter() - started)
        if not torch.equal(final, baseline):
            raise RuntimeError("Uninstrumented profile differs from measured baseline")
        del final
    peak = torch.cuda.max_memory_allocated()
    per_block = args.steps // (args.length // args.block_length)
    sampled_steps = sorted({0, per_block // 2, args.steps - 1})
    # Stage input transfers outside profiler regions; input collection is not part
    # of operator attribution. Timed full decoding above has NO profiling scopes.
    inputs = [(step, states[step].to(llm.device)) for step in sampled_steps]
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        with decoder_scopes(llm):
            for step, x in inputs:
                with torch.profiler.record_function(f"sampled_forward.step_{step}"):
                    output = forward_logits(llm, prepared, x)
                    del output
                prof.step()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(work / "baseline_trace.json"))
    events = prof.key_averages()
    (work / "baseline_operators.txt").write_text(events.table(sort_by="self_cuda_time_total", row_limit=50))
    scopes = [{"name": e.key, "calls": e.count, "cpu_total_us": e.cpu_time_total,
               "device_total_us": e.device_time_total}
              for e in events if e.key.startswith("decoder.")]
    operators = [{"name": e.key, "calls": e.count, "self_device_us": e.self_device_time_total}
                 for e in sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:50]]
    ids = baseline[0, prepared.prompt.shape[1]:].tolist()
    first_eos = ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in ids else None
    write_json(work / "baseline_profile.json", {
        "audio_preparation_seconds": preparation_seconds,
        "audio_preparation_median_seconds": statistics.median(preparation_seconds),
        "decoding_seconds": decoding_seconds,
        "decoding_median_seconds": statistics.median(decoding_seconds),
        "peak_allocated_bytes": peak, "sampled_forward_steps": sampled_steps,
        "audio_semantic_tokens": prepared.audio_count - prepared.acoustic_count,
        "audio_acoustic_tokens": prepared.acoustic_count,
        "prompt_tokens": prepared.prompt.shape[1], "answer_slots": args.length,
        "first_eos_answer_position": first_eos, "forwards": args.steps,
        "scopes": scopes, "operators": operators,
        "note": "Warmed, two repeats, batch 1; decoding timings have no measurement hooks. "
                "Audio preparation includes CPU feature extraction, transfer, encoder, adapters and prompt setup, "
                "but excludes loading weights and audio from disk. Fixed-budget decoding fills all slots, "
                "including positions after EOS. Traces cover only sampled baseline forwards and include "
                "profiling overhead. Dense operation times do not isolate audio token cost or predict "
                "achievable cache speedup; cached text-to-audio attention still costs compute."})
    print(f"  clean baseline decode: {statistics.median(decoding_seconds):.2f}s; "
          f"audio preparation: {statistics.median(preparation_seconds):.2f}s", flush=True)
