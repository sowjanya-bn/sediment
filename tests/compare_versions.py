import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from datetime import datetime

from extractor.extract import extract as extract_new
from extractor.core import extract as extract_old

TEST_INPUTS = [
    "I feel focused and excited about the work I’m doing today.",
    "I am not sure what I feel. I keep circling the same thoughts.",
    "I feel emotionally flat and disconnected today.",
    "I am overwhelmed but also strangely motivated.",
    "I cannot focus on anything today.",
    "Things are not terrible, just strangely muted.",
]

def run_comparison():
    results = []

    for text in TEST_INPUTS:
        old = extract_old(text)
        new = extract_new(text, backend="ollama", ollama_model="llama3.2")

        results.append({
            "input": text,
            "old": old,
            "new": new,
        })

    return results


def save_results(results):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"tests/results/comparison_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {filename}")


if __name__ == "__main__":
    results = run_comparison()
    save_results(results)


