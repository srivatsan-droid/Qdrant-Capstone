"""
data.py

Loads the scikit-learn 20 Newsgroups dataset and saves a sample of
(text, label, category) records to disk, so embedding/indexing scripts
downstream don't need to re-download or re-sample on every run.
"""

import pickle
import random
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

RANDOM_SEED = 42
SAMPLE_SIZE = 6000
OUTPUT_PATH = Path("data/newsgroups_sample.pkl")


def load_and_sample(sample_size: int = SAMPLE_SIZE, seed: int = RANDOM_SEED):
    """Download (or use sklearn's local cache of) 20 Newsgroups and return a
    random sample of {text, label, category} dicts, skipping empty docs."""
    newsgroups = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),  # strip metadata that would
        random_state=seed,                          # make classification "too easy"
    )

    texts = newsgroups.data
    labels = newsgroups.target
    target_names = newsgroups.target_names
    total = len(texts)

    if sample_size >= total:
        indices = list(range(total))
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(total), sample_size)

    documents = []
    for i in indices:
        text = texts[i].strip()
        if not text:
            continue
        label = int(labels[i])
        documents.append(
            {"text": text, "label": label, "category": target_names[label]}
        )

    return documents


def main():
    documents = load_and_sample()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(documents, f)

    categories = sorted(set(d["category"] for d in documents))
    print(f"Sampled {len(documents)} documents across {len(categories)} categories.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
