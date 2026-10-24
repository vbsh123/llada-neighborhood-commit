"""Load official DIFFA weights, avoiding upstream's hard-coded local paths."""
import gc
import json
from pathlib import Path
import sys

import torch

from .checkout import verify_checkout
from .decode import Prepared

DIFFA_REPO = "zhoujiaming777/DIFFA-2"
DIFFA_REVISION = "8aea6122bafb25578cb49d142c6ac1f8e83e3dea"
LLM_REPO = "GSAI-ML/LLaDA-8B-Instruct"
LLM_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"


def resolve_revision(repo, revision):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo, revision=revision).sha


def load_model(root, diffa_repo, diffa_revision, llm_repo, llm_revision,
               whisper_repo, whisper_revision, device="cuda", lora_alpha=16):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    source = verify_checkout(root)
    # Official checkout has no root .gitignore; imports must not dirty it.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    # Imported only after checking the exact source revision.
    from src.configuration_DIFFA import DIFFAConfig
    from src.modeling_DIFFA import DIFFAModel
    from src.fast_dllm.model.modeling_llada import LLaDAModelLM

    config_path = hf_hub_download(diffa_repo, "config.json", revision=diffa_revision)
    raw = json.loads(Path(config_path).read_text())
    config = DIFFAConfig(**raw)
    # The upstream config class overwrites these values. Restore checkpoint
    # values before constructing the connectors (also supports original DIFFA).
    config.whisper_model_id = whisper_repo
    config.prompt_size = raw.get("prompt_size", 64)
    tokenizer = AutoTokenizer.from_pretrained(llm_repo, revision=llm_revision)
    extractor = WhisperFeatureExtractor.from_pretrained(whisper_repo, revision=whisper_revision)
    model = DIFFAModel(config, tokenizer=tokenizer, stage=2)
    checkpoint_path = hf_hub_download(diffa_repo, "pytorch_model.bin", revision=diffa_revision)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError("Expected DIFFA tensor state dictionary")
    adapters = {k: v for k, v in state.items() if not k.startswith("llm_model.")}
    incompatible = model.load_state_dict(adapters, strict=False)
    required_missing = [k for k in incompatible.missing_keys
                        if k.startswith(("semantic_connector.", "acoustic_connector."))]
    if required_missing or incompatible.unexpected_keys:
        raise ValueError(f"Adapter mismatch: missing={required_missing}; unexpected={incompatible.unexpected_keys}")

    # Use HF's standard loader (supports safetensors) for the encoder, then put
    # weights into upstream's length-aware encoder class. Upstream also replaces
    # the encoder with the public Whisper weights after loading DIFFA.
    whisper = WhisperModel.from_pretrained(whisper_repo, revision=whisper_revision,
                                           torch_dtype=torch.float32)
    model.whisper_model.load_state_dict(whisper.encoder.state_dict(), strict=True)
    del whisper, adapters
    gc.collect()
    llm = LLaDAModelLM.from_pretrained(llm_repo, revision=llm_revision,
                                      torch_dtype=torch.bfloat16)
    llm_state = {k[len("llm_model."):]: v for k, v in state.items()
                 if k.startswith("llm_model.")}
    lora = {k: v for k, v in llm_state.items() if "lora_" in k}
    load_report = {"adapter_missing_noncritical": incompatible.missing_keys,
                   "llm_checkpoint_keys": len(llm_state), "lora_keys": len(lora)}
    if lora:
        from peft import LoraConfig, get_peft_model
        ranks = {v.shape[0] for k, v in lora.items() if ".lora_A." in k}
        if len(ranks) != 1:
            raise ValueError(f"Expected a single LoRA rank, found {ranks}")
        targets = sorted({k.split(".lora_")[0].split(".")[-1] for k in lora})
        llm = get_peft_model(llm, LoraConfig(r=ranks.pop(), lora_alpha=lora_alpha,
                                            target_modules=targets, lora_dropout=.05,
                                            bias="none", task_type="CAUSAL_LM"))
        converted = {k if k.startswith("base_model.model.") else "base_model.model." + k: v
                     for k, v in llm_state.items()}
        report = llm.load_state_dict(converted, strict=False)
        missing_lora = [k for k in report.missing_keys if "lora_" in k]
        if missing_lora or report.unexpected_keys:
            raise ValueError(f"LoRA mismatch: {missing_lora}, {report.unexpected_keys}")
        llm = llm.merge_and_unload()
        load_report["lora_alpha"] = lora_alpha
    elif llm_state:
        llm.load_state_dict(llm_state, strict=True)
    del state, llm_state, lora
    gc.collect()
    model.llm_model = llm
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer, extractor, load_report


@torch.no_grad()
def prepare(model, tokenizer, extractor, audio, instruction, length,
            system_prompt="You are a helpful voice assistant."):
    from src.dataloader import build_chat_prompt

    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt",
                       return_attention_mask=True, truncation=False)
    inputs = inputs.to(model.llm_model.device)
    speech, valid = model.get_speech_features(inputs.input_features, inputs.attention_mask, stage=2)
    acoustic_count = model.acoustic_connector.layer_prompts[0].shape[1]
    if speech.shape[0] != 1 or not valid.all() or valid.shape[1] != speech.shape[1]:
        raise ValueError(f"Expected one clip with only valid, unpadded projected audio tokens; "
                         f"speech={tuple(speech.shape)}, valid={tuple(valid.shape)}, "
                         f"valid_count={int(valid.sum())}, acoustic={acoustic_count}")
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "This is the audio: "}]
    prefix = build_chat_prompt(messages, add_generation_prompt=False)[:-10]
    messages[-1]["content"] = "This is the audio: <audio>. " + instruction
    text = build_chat_prompt(messages, add_generation_prompt=True)
    # Reproduce upstream's insertion offset, but refuse silent truncation.
    prefix_ids = tokenizer(prefix, return_tensors="pt")
    encoded = tokenizer(text, return_tensors="pt")
    insertion = int(prefix_ids.attention_mask.sum()) - 1
    prompt = encoded.input_ids.to(model.llm_model.device)
    maximum = model.llm_model.config.max_sequence_length
    if prompt.shape[1] + speech.shape[1] + length > maximum:
        raise ValueError(f"Combined audio/prompt/answer exceeds {maximum} positions")
    mask_id = model.llm_model.config.mask_token_id
    if (prompt == mask_id).any() or not 0 <= insertion <= prompt.shape[1]:
        raise ValueError("Invalid prompt MASK or insertion position")
    return Prepared(prompt, speech.to(torch.bfloat16), insertion, acoustic_count, mask_id)
