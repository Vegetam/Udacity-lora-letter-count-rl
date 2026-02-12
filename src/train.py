"""GRPO training for the Udacity letter-counting project.

Task: "How many of the letter 'g' are there in the word 'engage'?"
Uses GRPOTrainer from trl with Qwen2.5-1.5B-Instruct + LoRA.

Uses fp16 — the 1.5B model is ~3 GB, leaving plenty of room for
GRPO generation + training on 8 GB VRAM GPUs.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, get_peft_model

from .rewards import (
    format_reward_func,
    numbering_reward_func,
    spelling_reward_func,
    counting_reward_func,
    correct_answer_reward_func,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "mean_correctness_over_time.csv"
BASELINE_PATH = PROJECT_ROOT / "outputs" / "logs" / "baseline_output.txt"
FINAL_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "final_lora_adapter"

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# LoRA rank 32 — one of the valid values (8, 16, 32, 64, 128).
# 32 gives good capacity while staying memory-friendly on 8 GB VRAM.
LORA_RANK = 32

# Target all key linear layers in attention + MLP blocks.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── System prompt with CoT + two-shot example ──────────────────────────────
SYSTEM_PROMPT = """\
Respond in the following format:
<reasoning>
Counting the number of [letter_to_count]'s in the word [word]
1. [first letter] - [count of requested letter so far] so far
2. [second letter] - [count of requested letter so far] so far
...
</reasoning>
<answer>
[number]
</answer>"""

# ── Word list ───────────────────────────────────────────────────────────────
ALL_WORDS = [
    "idea", "glow", "rust", "maze", "echo",
    "wisp", "veto", "lush", "gaze", "knit",
    "crisp", "lunar", "fable", "quest", "verge",
    "brawn", "elude", "aisle", "ember", "crave",
    "mosaic", "velvet", "sphinx", "radius", "summit",
    "banner", "cipher", "mantle", "scarab", "expose",
    "lantern", "enchant", "torrent", "capture", "orchard",
    "eclipse", "triumph", "absolve", "whistle", "resolve",
]


def generate_records():
    """
    For each word, yield one record per unique letter in the word,
    plus a couple of letters NOT in the word (count = 0).
    This matches the reference project's dataset creation.
    """
    for word in ALL_WORDS:
        for letter in sorted(set(word)):
            yield {"words": word, "letters": letter, "counts": word.count(letter)}
        # Also include some letters that are NOT in the word
        num_absent = int(len(word) // 7 + 1)
        random.seed(hash(word))
        all_letters = list("abcdefghijklmnopqrstuvwxyz")
        random.shuffle(all_letters)
        for letter in all_letters:
            if letter not in word:
                yield {"words": word, "letters": letter, "counts": 0}
                num_absent -= 1
            if num_absent == 0:
                break


def build_dataset() -> Dataset:
    """Create the GRPO training dataset with chat-message prompts."""
    ds = Dataset.from_generator(generate_records)
    ds = ds.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": 'How many of the letter "{}" are there in the word "{}"'.format(
                        x["letters"], x["words"]
                    ),
                },
            ],
        }
    )
    return ds


# ── Baseline demonstration ──────────────────────────────────────────────────

@torch.no_grad()
def demonstrate_baseline(model, tokenizer) -> None:
    """Generate baseline (pre-training) responses and save them."""
    print("\n" + "=" * 70)
    print("BASELINE (before GRPO training)")
    print("=" * 70)

    test_cases = [
        ("engage", "g"),
        ("banana", "a"),
        ("strawberry", "r"),
    ]
    results: List[str] = []

    for word, letter in test_cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'How many of the letter "{letter}" are there in the word "{word}"'},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        response = full_text[len(prompt_decoded):].strip()
        block = f'Q: How many "{letter}" in "{word}"?  (true={word.count(letter)})\n{response}'
        results.append(block)
        print(f"\n{block}")

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        f"System Prompt:\n{SYSTEM_PROMPT}\n\n" + "\n---\n".join(results) + "\n",
        encoding="utf-8",
    )
    print(f"\nBaseline saved to {BASELINE_PATH}")


# ── Logging callback ────────────────────────────────────────────────────────

class CorrectnessLogger(TrainerCallback):
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step == 0:
            return
        key = "rewards/correct_answer_reward_func/mean"
        if key in logs:
            self.rows.append({
                "step": int(state.global_step),
                "mean_correctness_reward": float(logs[key]),
            })
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.rows).to_csv(LOG_PATH, index=False)
            print(f"  Step {state.global_step:3d}: correctness={logs[key]:+.3f}")


# ── Main training function ──────────────────────────────────────────────────

def train(max_steps: int = 80) -> None:
    print("=" * 70)
    print("GRPO TRAINING — Udacity Letter-Counting Project")
    print("=" * 70)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 2. Model (~3 GB in fp16 — fits easily on 8 GB VRAM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\nLoading {MODEL_NAME} in {'fp16' if device == 'cuda' else 'fp32'}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    # 3. LoRA adapter
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.1,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Baseline demo
    demonstrate_baseline(model, tokenizer)

    # 5. Dataset
    dataset = build_dataset()
    print(f"\nDataset: {len(dataset)} records")

    # 6. GRPO training config
    #    Tuned for 8 GB VRAM (RTX 5060) with 1.5B fp16:
    #    - batch_size=1, num_generations=2 → 2 forward passes per step
    #    - gradient_accumulation_steps=4  → effective batch = 4
    training_args = GRPOConfig(
        output_dir=str(PROJECT_ROOT / "outputs" / "checkpoints"),
        learning_rate=5e-6,
        beta=0.04,
        per_device_train_batch_size=1,
        num_generations=2,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        max_completion_length=200,
        logging_steps=1,
        save_steps=40,
        save_total_limit=2,
        fp16=(device == "cuda"),
        bf16=False,
        report_to=[],
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
    )

    # 7. GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            numbering_reward_func,
            spelling_reward_func,
            counting_reward_func,
            format_reward_func,
            correct_answer_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.add_callback(CorrectnessLogger())

    # 8. Train
    print(f"\nStarting GRPO training for {max_steps} steps ...")
    print(f"  batch={training_args.per_device_train_batch_size}, "
          f"num_gen={training_args.num_generations}, "
          f"grad_accum={training_args.gradient_accumulation_steps}")
    trainer.train()

    # 9. Save adapter
    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    print(f"\nLoRA adapter saved to: {FINAL_ADAPTER_DIR}")

    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        print("\nReward progression:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    train()
