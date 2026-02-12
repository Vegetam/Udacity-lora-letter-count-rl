"""Reward functions for the GRPO letter-counting project.

Task: "How many of the letter 'g' are there in the word 'engage'?"

Expected completion format produced by the model:
<reasoning>
Counting the number of g's in the word engage
1. e - 0 so far
2. n - 0 so far
3. g - 1 so far
4. a - 1 so far
5. g - 2 so far
6. e - 2 so far
</reasoning>
<answer>
2
</answer>

GRPOTrainer passes dataset columns as kwargs to reward functions.
Completions are message lists: [ {"role": "assistant", "content": "..."} ]
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ── Parsing helpers ─────────────────────────────────────────────────────────

def extract_xml_answer(text: str) -> str:
    """Extract content inside <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_letter_numbering(response: str) -> list[int]:
    """Extract the line numbers from numbered lines like '1. e - 0 so far'."""
    matches = re.findall(r"\n(\d+)\. [a-z]", response)
    return [int(m) for m in matches] if matches else []


def extract_spelling(response: str) -> str:
    """Extract the letters listed on numbered lines."""
    matches = re.findall(r"\n\d+\. ([a-z])", response, flags=re.IGNORECASE)
    return "".join(matches) if matches else ""


def get_resp_letters_and_counts(response: str) -> list[tuple[str, str]]:
    """Extract (letter, running_count) pairs from numbered lines."""
    matches = re.findall(r"\n(\d+)\. ([a-z])\D*(\d+)", response, flags=re.IGNORECASE)
    return [(letter, count) for _, letter, count in matches] if matches else []


def _get_content(completion) -> str:
    """Get text content from a completion (handles message-list or string)."""
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        if completion and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion) if completion else ""


# ── Reward functions for GRPOTrainer ────────────────────────────────────────

def numbering_reward_func(completions, words, **kwargs) -> list[float]:
    """
    Reward correct sequential numbering of each letter.
    +1.0 per correctly-numbered line, -1.0 per wrong number.
    Extra lines beyond the word length are also penalised.
    Normalised by word length.
    """
    res = []
    for completion, word in zip(completions, words):
        response = _get_content(completion)
        reward = 0
        for ix, spell_number in enumerate(extract_letter_numbering(response)):
            line_number = ix + 1
            if spell_number == line_number:
                reward += 1.0
            else:
                reward -= 1.0
            if line_number > len(word):
                reward -= (line_number - len(word))
        res.append(reward / max(1, len(word)))
    return res


def spelling_reward_func(completions, words, **kwargs) -> list[float]:
    """
    Reward when the listed letters match the word exactly.
    +2.0 for perfect match; penalties for wrong / extra / missing letters.
    """
    res = []
    for completion, word in zip(completions, words):
        response = _get_content(completion)
        spelled = extract_spelling(response)
        reward = 0.0
        reward += 2.0 if spelled == word else 0.0
        reward -= abs(len(word) - len(spelled)) / max(1, len(word))
        reward -= sum((Counter(spelled) - Counter(word)).values()) / max(1, len(word))
        reward -= sum((Counter(word) - Counter(spelled)).values()) / max(1, len(word))
        res.append(reward)
    return res


def counting_reward_func(completions, letters, **kwargs) -> list[float]:
    """
    Reward an accurate running count of the target letter.
    At each numbered line, check if the running total is correct.
    +1.0 per accurate step, -1.0 per inaccurate step.
    Normalised by the number of lines parsed.
    """
    res = []
    for completion, letter in zip(completions, letters):
        response = _get_content(completion)
        letters_and_counts = get_resp_letters_and_counts(response)
        if not letters_and_counts:
            res.append(-1.0)
            continue
        reward = 0
        actual_count = 0
        for resp_letter, resp_count in letters_and_counts:
            if letter == resp_letter.lower():
                actual_count += 1
            if actual_count == int(resp_count):
                reward += 1.0
            else:
                reward -= 1.0
        res.append(reward / max(1, len(letters_and_counts)))
    return res


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    +0.5 if <reasoning>...</reasoning><answer>...</answer> format is used.
    +0.5 if the extracted answer is a digit.
    """
    res = []
    for completion in completions:
        response = _get_content(completion)
        reward = 0.0
        if re.match(
            r"\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
            response,
            flags=re.MULTILINE | re.DOTALL,
        ):
            reward += 0.5
        extracted = extract_xml_answer(response)
        if extracted.isdigit():
            reward += 0.5
        res.append(reward)
    return res


def correct_answer_reward_func(prompts, completions, counts, **kwargs) -> list[float]:
    """
    +2.0 if the extracted answer matches the true count.
     0.0 otherwise.
    """
    res = []
    for completion, count in zip(completions, counts):
        response = _get_content(completion)
        extracted = extract_xml_answer(response)
        res.append(2.0 if str(extracted) == str(count) else 0.0)
    return res


# ── Convenience helper for evaluate.py ──────────────────────────────────────

@dataclass
class RewardBreakdown:
    formatting: float
    numbering: float
    spelling: float
    counting: float
    correctness: float

    @property
    def total(self) -> float:
        return self.formatting + self.numbering + self.spelling + self.counting + self.correctness


def reward_single(word: str, letter: str, count: int, completion: str) -> RewardBreakdown:
    """Score a single completion. Used by evaluate.py and notebooks."""
    wrapped = [completion]
    return RewardBreakdown(
        formatting=format_reward_func(wrapped)[0],
        numbering=numbering_reward_func(wrapped, words=[word])[0],
        spelling=spelling_reward_func(wrapped, words=[word])[0],
        counting=counting_reward_func(wrapped, letters=[letter])[0],
        correctness=correct_answer_reward_func(
            prompts=[""], completions=wrapped, counts=[count]
        )[0],
    )
