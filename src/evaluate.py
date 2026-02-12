"""Final comparison: pretrained (base) vs GRPO fine-tuned model."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .rewards import reward_single
from .train import SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "final_lora_adapter"
EVAL_OUTPUT = PROJECT_ROOT / "outputs" / "logs" / "evaluation_output.txt"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _load_model(adapter: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if adapter:
        if not (ADAPTER_DIR / "adapter_config.json").exists():
            raise FileNotFoundError(f"No adapter at {ADAPTER_DIR}. Run training first.")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    return model, tokenizer


@torch.no_grad()
def _generate(model, tokenizer, word: str, letter: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'How many of the letter "{letter}" are there in the word "{word}"'},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full[len(prompt_decoded):].strip()


def compare_models(word: str, letter: str):
    """Compare base vs fine-tuned on a specific letter-counting question."""
    true_count = word.count(letter)
    print(f"{'=' * 60}")
    print(f'  How many "{letter}" in "{word}"?  (true count: {true_count})')
    print(f"{'=' * 60}")

    # Base model
    base_model, tok = _load_model(adapter=False)
    base_out = _generate(base_model, tok, word, letter)
    base_r = reward_single(word, letter, true_count, base_out)
    print("\n--- OLD (pretrained, no fine-tuning) ---")
    print(base_out)
    print(f"Rewards: {base_r}")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Fine-tuned model
    ft_model, tok = _load_model(adapter=True)
    ft_out = _generate(ft_model, tok, word, letter)
    ft_r = reward_single(word, letter, true_count, ft_out)
    print("\n--- NEW (GRPO fine-tuned) ---")
    print(ft_out)
    print(f"Rewards: {ft_r}")

    return base_out, ft_out


def compare_general_knowledge(question: str):
    """Check for catastrophic forgetting with a general knowledge question."""
    print(f"{'=' * 60}")
    print(f"  Catastrophic forgetting check: {question}")
    print(f"{'=' * 60}")

    for label, use_adapter in [("OLD (pretrained)", False), ("NEW (fine-tuned)", True)]:
        model, tok = _load_model(adapter=use_adapter)
        messages = [{"role": "user", "content": question}]
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
        prompt_decoded = tok.decode(inputs["input_ids"][0], skip_special_tokens=True)
        answer = tok.decode(out[0], skip_special_tokens=True)[len(prompt_decoded):].strip()
        print(f"\n--- {label} ---")
        print(answer)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Capture all output to both console and file
    buf = io.StringIO()

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    tee = Tee(sys.stdout, buf)
    old_stdout = sys.stdout
    sys.stdout = tee

    # Test on words from the training set
    compare_models("banner", "n")
    print()
    compare_models("eclipse", "e")
    print()
    # Catastrophic forgetting check
    compare_general_knowledge("What is the capital of France?")

    sys.stdout = old_stdout

    # Save to file
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nEvaluation saved to: {EVAL_OUTPUT}")
