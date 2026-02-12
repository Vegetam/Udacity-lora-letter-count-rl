"""
Diagnostic: verify reward functions score correct samples higher than incorrect.
"""
from src.utils import read_jsonl
from src.rewards import reward_single


def test_rewards():
    correct = read_jsonl("data/samples_correct.jsonl")
    incorrect = read_jsonl("data/samples_incorrect.jsonl")

    print("=== Correct samples ===")
    for row in correct:
        r = reward_single(row["word"], row["letter"], row["count"], row["completion"])
        print(f"  \"{row['letter']}\" in \"{row['word']}\"  rewards={r}  total={r.total:.1f}")

    print("\n=== Incorrect samples ===")
    for row in incorrect:
        r = reward_single(row["word"], row["letter"], row["count"], row["completion"])
        print(f"  \"{row['letter']}\" in \"{row['word']}\"  rewards={r}  total={r.total:.1f}")


if __name__ == "__main__":
    test_rewards()
