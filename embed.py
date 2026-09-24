"""
embed.py

Embeds the sampled 20 Newsgroups documents (produced by data.py) and the 5
test queries (from queries.md) with a local sentence-transformers model, and
writes vectors + metadata to disk so later steps don't need to re-embed.
"""

import pickle
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/newsgroups_sample.pkl")
QUERIES_PATH = Path("queries.md")
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast enough on CPU

DOC_VECTORS_PATH = Path("data/doc_vectors.npy")
DOC_METADATA_PATH = Path("data/doc_metadata.pkl")
QUERY_VECTORS_PATH = Path("data/query_vectors.npy")
QUERY_TEXT_PATH = Path("data/query_texts.pkl")


def load_documents():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found — run `python data.py` first.")
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def load_queries():
    """Pull the numbered, quoted query lines out of queries.md, e.g.
    `1. "a question about a graphics card driver"`."""
    text = QUERIES_PATH.read_text(encoding="utf-8")
    matches = re.findall(r'^\d+\.\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(
            "Could not find numbered, quoted queries in queries.md. "
            'Expected lines like: 1. "some query text"'
        )
    return matches


def main():
    print(f"Loading model '{MODEL_NAME}' (downloads once, cached after)...")
    model = SentenceTransformer(MODEL_NAME)

    documents = load_documents()
    texts = [d["text"] for d in documents]

    print(f"Embedding {len(texts)} documents...")
    doc_vectors = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )

    queries = load_queries()
    print(f"Embedding {len(queries)} test queries...")
    query_vectors = model.encode(queries, convert_to_numpy=True)

    Path("data").mkdir(exist_ok=True)
    np.save(DOC_VECTORS_PATH, doc_vectors)
    with open(DOC_METADATA_PATH, "wb") as f:
        pickle.dump(documents, f)
    np.save(QUERY_VECTORS_PATH, query_vectors)
    with open(QUERY_TEXT_PATH, "wb") as f:
        pickle.dump(queries, f)

    print(f"Saved {doc_vectors.shape[0]} document vectors "
          f"(dim={doc_vectors.shape[1]}) -> {DOC_VECTORS_PATH}")
    print(f"Saved {query_vectors.shape[0]} query vectors -> {QUERY_VECTORS_PATH}")


if __name__ == "__main__":
    main()
