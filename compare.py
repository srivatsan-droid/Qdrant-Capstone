"""
compare.py

Runs the 5 test queries through every method covered in the project:
  - Part 2: three distance metrics (Cosine, Euclidean, Dot) on identical data
  - Part 3: exact search vs. HNSW (default and under-tuned) at several ef
  - Part 4: the from-scratch IVF index at several nprobe values
Writes results to results/*.json and prints progress as it goes, so a single
run of `python compare.py` is the live demo for the review session.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams

from ivf import load_index, search_ivf

QDRANT_URL = "http://localhost:6333"
RESULTS_DIR = Path("results")

QUERY_VECTORS_PATH = Path("data/query_vectors.npy")
QUERY_TEXT_PATH = Path("data/query_texts.pkl")
DOC_METADATA_PATH = Path("data/doc_metadata.pkl")

DISTANCE_COLLECTIONS = ["newsgroups_cosine", "newsgroups_euclidean", "newsgroups_dot"]
HNSW_DEFAULT_COLLECTION = "newsgroups_hnsw_default"
HNSW_UNDERTUNED_COLLECTION = "newsgroups_hnsw_undertuned"
EF_VALUES = [16, 64, 128]
NPROBE_VALUES = [1, 8]
TOP_K = 5


def load_queries():
    if not QUERY_VECTORS_PATH.exists():
        raise FileNotFoundError(f"{QUERY_VECTORS_PATH} not found — run `python embed.py` first.")
    query_vectors = np.load(QUERY_VECTORS_PATH)
    with open(QUERY_TEXT_PATH, "rb") as f:
        query_texts = pickle.load(f)
    return query_texts, query_vectors


def load_metadata():
    with open(DOC_METADATA_PATH, "rb") as f:
        return pickle.load(f)


def hit_ids(scored_points):
    return [p.id for p in scored_points]


def overlap(ids_a, ids_b):
    return len(set(ids_a) & set(ids_b)) / len(ids_a) if ids_a else 0.0


# ---------------------------------------------------------------------------
# Part 2: distance metrics
# ---------------------------------------------------------------------------

def run_distance_metric_comparison(client, query_texts, query_vectors, metadata):
    print("\n=== Part 2: distance metrics (Cosine vs Euclidean vs Dot) ===")
    output = []
    for q_text, q_vec in zip(query_texts, query_vectors):
        row = {"query": q_text, "results": {}}
        for collection in DISTANCE_COLLECTIONS:
            hits = client.query_points(
                collection_name=collection, query=q_vec.tolist(), limit=TOP_K
            ).points
            row["results"][collection] = [
                {"id": h.id, "score": h.score, "category": metadata[h.id]["category"]}
                for h in hits
            ]
            top = hits[0]
            print(f"  [{collection}] '{q_text[:40]}...' -> top id {top.id} "
                  f"({metadata[top.id]['category']}, score={top.score:.4f})")
        output.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "distance_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved -> {RESULTS_DIR / 'distance_metrics.json'}")
    return output


# ---------------------------------------------------------------------------
# Part 3: exact search vs HNSW (default and under-tuned) across ef values
# ---------------------------------------------------------------------------

def timed_search(client, collection, query_vector, exact=False, ef=None):
    if exact:
        params = SearchParams(exact=True)
    elif ef is not None:
        params = SearchParams(hnsw_ef=ef, exact=False)
    else:
        params = None

    start = time.perf_counter()
    hits = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=TOP_K,
        search_params=params,
    ).points
    elapsed_ms = (time.perf_counter() - start) * 1000
    return hits, elapsed_ms


def run_hnsw_comparison(client, query_texts, query_vectors):
    print("\n=== Part 3: exact search vs HNSW (default vs under-tuned) ===")
    output = []

    for q_text, q_vec in zip(query_texts, query_vectors):
        exact_hits, exact_ms = timed_search(client, HNSW_DEFAULT_COLLECTION, q_vec, exact=True)
        exact_ids = hit_ids(exact_hits)
        print(f"  [exact] '{q_text[:40]}...' -> {exact_ms:.2f}ms, top id {exact_ids[0]}")

        row = {
            "query": q_text,
            "exact": {"ids": exact_ids, "latency_ms": exact_ms},
            "hnsw_default": {},
            "hnsw_undertuned": {},
        }

        for label, collection in [
            ("hnsw_default", HNSW_DEFAULT_COLLECTION),
            ("hnsw_undertuned", HNSW_UNDERTUNED_COLLECTION),
        ]:
            for ef in EF_VALUES:
                hits, latency_ms = timed_search(client, collection, q_vec, ef=ef)
                ids = hit_ids(hits)
                ov = overlap(exact_ids, ids)
                row[label][str(ef)] = {
                    "ids": ids,
                    "latency_ms": latency_ms,
                    "overlap_with_exact": ov,
                }
                print(f"  [{label} ef={ef}] '{q_text[:30]}...' -> "
                      f"{latency_ms:.2f}ms, overlap={ov:.2f}")

        output.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "hnsw.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved -> {RESULTS_DIR / 'hnsw.json'}")
    return output


# ---------------------------------------------------------------------------
# Part 4: from-scratch IVF index at different nprobe values
# ---------------------------------------------------------------------------

def run_ivf_comparison(query_texts, query_vectors, exact_ids_by_query):
    print("\n=== Part 4: from-scratch IVF index ===")
    ivf_index = load_index()
    output = []

    for q_text, q_vec, exact_ids in zip(query_texts, query_vectors, exact_ids_by_query):
        row = {"query": q_text, "results": {}}
        for nprobe in NPROBE_VALUES:
            start = time.perf_counter()
            hits = search_ivf(q_vec, ivf_index, nprobe=nprobe, k=TOP_K)
            latency_ms = (time.perf_counter() - start) * 1000
            ids = [h[0] for h in hits]
            ov = overlap(exact_ids, ids)
            row["results"][str(nprobe)] = {
                "ids": ids,
                "scores": [h[1] for h in hits],
                "latency_ms": latency_ms,
                "overlap_with_exact": ov,
            }
            print(f"  [ivf nprobe={nprobe}] '{q_text[:30]}...' -> "
                  f"{latency_ms:.2f}ms, overlap={ov:.2f}")
        output.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "ivf.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved -> {RESULTS_DIR / 'ivf.json'}")
    return output


def main():
    client = QdrantClient(url=QDRANT_URL)
    query_texts, query_vectors = load_queries()
    metadata = load_metadata()

    run_distance_metric_comparison(client, query_texts, query_vectors, metadata)
    hnsw_results = run_hnsw_comparison(client, query_texts, query_vectors)
    exact_ids_by_query = [row["exact"]["ids"] for row in hnsw_results]
    run_ivf_comparison(query_texts, query_vectors, exact_ids_by_query)

    print("\nAll comparisons complete. Raw data is in results/.")
    print("Next: fill in comparison.md using results/*.json.")


if __name__ == "__main__":
    main()