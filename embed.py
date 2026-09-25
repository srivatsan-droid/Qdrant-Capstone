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


def load_model():
    """Load the sentence-transformers model, with its built-in
    L2-normalization stripped out.

    `all-MiniLM-L6-v2` (like most of the `all-*` sentence-transformers
    models) ships with a `Normalize` module baked into its pipeline, on top
    of the transformer + mean-pooling. That makes *every* embedding it
    produces unit-length, no matter what you pass to `.encode()`.

    That matters a lot for Part 2: if every document vector and every query
    vector has norm 1, then dot product and cosine similarity are always
    the *same ranking* (dot = cosine * ||query||, and ||query|| is constant
    for a given query), and Euclidean distance is a strictly monotonic
    function of cosine similarity for unit vectors
    (d^2 = 2 - 2*cos). So with unit vectors, cosine / Euclidean / dot can
    *never* disagree on ranking, for any query -- there would be nothing to
    find for Part 2, no matter how many queries you tried.

    Removing the trailing Normalize module keeps the same transformer +
    pooling (so it's still "real" all-MiniLM-L6-v2 semantics), but lets
    embeddings keep their raw, document-dependent magnitude, so dot product
    can actually diverge from cosine.
    """
    model = SentenceTransformer(MODEL_NAME)
    last_key = list(model._modules.keys())[-1]
    if type(model._modules[last_key]).__name__ == "Normalize":
        print(
            "Detected a built-in Normalize module on "
            f"'{MODEL_NAME}' -- removing it so embeddings keep their "
            "raw magnitude (needed for Part 2 to be meaningful)."
        )
        del model._modules[last_key]
    return model


def main():
    print(f"Loading model '{MODEL_NAME}' (downloads once, cached after)...")
    model = load_model()

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
