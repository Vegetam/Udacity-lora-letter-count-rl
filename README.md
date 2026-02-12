# LoRA Letter-Counting Project (Baseline + Rewards + Training)

This repo is structured to satisfy the rubric requirements:

## 1) Model Setup (LoRA via PEFT)
- LoRA implemented with PEFT in `src/lora_setup.py`
- Hyperparameters explicitly set:
  - `lora_rank`: **64** (valid: 8,16,32,64,128)
  - `target_modules`: includes **q_proj, k_proj, v_proj, o_proj** and also tries **gate_proj, up_proj, down_proj**
- The code instantiates a trainable PEFT model before training (see notebook `03_lora_setup.ipynb`)

## 2) Baseline Prompting (One-shot CoT)
Notebook: `notebooks/01_baseline_prompting.ipynb`
- Demonstrates a **single-example** Chain-of-Thought prompt for the letter-counting task.
- Shows at least one concrete input/output where step-by-step reasoning is visible.

## 3) Reward Design & Validation
Code: `src/rewards.py`
Rewards include:
- numbering
- spelling
- counting
- formatting
- correctness (strict gate)

Notebook: `notebooks/02_reward_functions.ipynb`
- Validates rewards score a correct sample higher than an incorrect sample.

## 4) Training & Monitoring
Training script: `src/train.py`
- Runs a **longer** training run (`num_train_epochs=3`) with LoRA.
- Logs **mean correctness reward over time** to `outputs/logs/mean_correctness_over_time.csv`

Notebook: `notebooks/04_training_monitoring.ipynb`
- Loads the CSV and plots reward trend.

## 5) Final Comparison (Base vs Fine-tuned)
Notebook: `notebooks/05_final_comparison.ipynb` and script `src/evaluate.py`
- Runs the **base** model and the **fine-tuned (LoRA)** model on at least one example from `data/dataset.jsonl`.

---

## Quickstart

```bash
pip install -r requirements.txt
# baseline (prompting only)
python -m src.prompting_baseline

# train LoRA
python -m src.train

# compare base vs fine-tuned
python -m src.evaluate
```

## Dataset
- `data/dataset.jsonl`: prompt/completion pairs for the letter-counting task
- `data/samples_correct.jsonl` and `data/samples_incorrect.jsonl`: used for reward validation
