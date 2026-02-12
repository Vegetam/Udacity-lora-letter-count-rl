"""Baseline prompting for the letter-counting task.

Uses Qwen2.5-1.5B-Instruct WITHOUT any fine-tuning to show baseline
performance on: "How many of letter X are in word Y?"
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train import SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_base_model():
    """Load the pretrained model without any adapter."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, word: str, letter: str, max_new_tokens: int = 200) -> str:
    """Generate a single response for the letter-counting task."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'How many of the letter "{letter}" are there in the word "{word}"'},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full[len(prompt_decoded):].strip()


def run_baseline(word: str, letter: str) -> str:
    """Convenience: load model, generate, return response."""
    model, tokenizer = load_base_model()
    return generate(model, tokenizer, word, letter)


if __name__ == "__main__":
    model, tokenizer = load_base_model()
    for word, letter in [("engage", "g"), ("banana", "a"), ("strawberry", "r")]:
        true_count = word.count(letter)
        print(f'=== How many "{letter}" in "{word}"? (true: {true_count}) ===')
        print(generate(model, tokenizer, word, letter))
        print()
