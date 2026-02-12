from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class LoraSetup:
    # Use Qwen2.5-3B-Instruct as specified in the project instructions.
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # lora_rank=32 chosen as a good balance between model capacity and memory.
    # Higher rank (e.g. 64, 128) captures more information but uses more VRAM;
    # lower rank (e.g. 8, 16) is more memory-efficient but less expressive.
    # 32 gives good capacity while staying memory-friendly on consumer GPUs.
    lora_rank: int = 32  # must be one of 8, 16, 32, 64, 128

    # lora_alpha = 2 * lora_rank is a common heuristic that keeps the
    # effective learning rate for LoRA weights stable.
    lora_alpha: int = 64

    lora_dropout: float = 0.05

    # Target all key linear layers in both the attention block AND the MLP block.
    # Attention: q_proj, k_proj, v_proj (queries/keys/values) + o_proj (output projection)
    # MLP: gate_proj, up_proj, down_proj (the feed-forward network)
    # Targeting all of these gives the adapter maximum expressiveness.
    target_modules_requested: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # 4-bit quantization to fit Qwen2.5-3B on an 8 GB GPU (RTX 5060).
    # This reduces model weights from ~6 GB (fp16) to ~2 GB (4-bit).
    use_4bit: bool = True


def _filter_existing_modules(model, requested: Tuple[str, ...]) -> List[str]:
    """Keep only target module names that actually exist in the model."""
    existing = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in requested:
            existing.add(leaf)
    return [m for m in requested if m in existing]


def build_trainable_lora_model(cfg: LoraSetup):
    """Instantiate a trainable LoRA model. Returns (model, tokenizer, lora_config)."""
    if cfg.lora_rank not in (8, 16, 32, 64, 128):
        raise ValueError("lora_rank must be one of 8, 16, 32, 64, 128")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for correct batch generation in decoder-only models
    tokenizer.padding_side = "left"

    # Quantization config for 4-bit loading (QLoRA)
    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = _filter_existing_modules(model, cfg.target_modules_requested)
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer, lora_config
