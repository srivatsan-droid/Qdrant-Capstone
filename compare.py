"""
compare.py

Runs the 5 test queries through every method covered in the project:
  - Part 2: three distance metrics (Cosine, Euclidean, Dot) on identical data
  - Part 3: exact search vs. HNSW (default and under-tuned) at several ef
  - Part 4: the from-scratch IVF index at several nprobe values
Writes results to results/*.json and prints progress as it goes, so a single
run of `python compare.py` is the live demo for the review session.
"""

import argparse
import json
import platform
import statistics
from importlib.metadata import version
import pickle
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams

from ivf import load_index, search_ivf
from qdrant_setup import wait_for_indexes
from report import write_report

QDRANT_URL = "http://localhost:6333"
RESULTS_DIR = Path("results")

QUERY_VECTORS_PATH = Path("data/query_vectors.npy")
QUERY_TEXT_PATH = Path("data/query_texts.pkl")
DOC_METADATA_PATH = Path("data/doc_metadata.pkl")

DISTANCE_COLLECTIONS = ["newsgroups_cosine", "newsgroups_euclidean", "newsgroups_dot"]
HNSW_DEFAULT_COLLECTION = "newsgroups_hnsw_default"
HNSW_UNDERTUNED_COLLECTION = "newsgroups_hnsw_undertuned"
EF_VALUES = [16, 64, 128]
REPEATS = 20
WARMUPS = 3
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
                collection_name=collection, query=q_vec.tolist(), limit=TOP_K,
                search_params=SearchParams(exact=True), with_payload=False
            ).points
            row["results"][collection] = [
                {"id": h.id, "score": h.score, "category": metadata[h.id]["category"]}
                for h in hits
            ]
            print(f"  {collection} top-5: {[(h.id, round(h.score, 6)) for h in hits]}")
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

    return measure(lambda: client.query_points(
        collection_name=collection, query=query_vector.tolist(), limit=TOP_K,
        search_params=params, with_payload=False,
    ).points)


def measure(operation):
    """Warm up, then average repeated end-to-end durations for one query."""
    for _ in range(WARMUPS):
        operation()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        hits = operation()
        samples.append((time.perf_counter() - start) * 1000)
    return hits, statistics.mean(samples)


def run_hnsw_comparison(client, query_texts, query_vectors):
    print("\n=== Part 3: exact search vs HNSW (default vs under-tuned) ===")
    output = []

    for q_text, q_vec in zip(query_texts, query_vectors):
        exact_hits, exact_ms = timed_search(client, HNSW_DEFAULT_COLLECTION, q_vec, exact=True)
        exact_ids = hit_ids(exact_hits)
        print(f"  exact top-5: {[(h.id, round(h.score, 6)) for h in exact_hits]}")
        print(f"  [exact] '{q_text[:40]}...' -> {exact_ms:.2f}ms, top id {exact_ids[0]}")

        row = {
            "query": q_text,
            "exact": {"ids": exact_ids, "scores": [h.score for h in exact_hits], "latency_ms": exact_ms},
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
                print(f"  {label} ef={ef} top-5: {[(h.id, round(h.score, 6)) for h in hits]}")
                ov = overlap(exact_ids, ids)
                row[label][str(ef)] = {
                    "ids": ids,
                    "scores": [h.score for h in hits],
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
            hits, latency_ms = measure(lambda: search_ivf(q_vec, ivf_index, nprobe=nprobe, k=TOP_K))
            print(f"  IVF nprobe={nprobe} top-5: {hits}")
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
    global REPEATS, RESULTS_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=int, help="One-based query number for live demo")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    REPEATS = args.repeats
    client = QdrantClient(url=QDRANT_URL)
    query_texts, query_vectors = load_queries()
    metadata = load_metadata()
    if len(query_texts) != 5 or len(query_vectors) != 5:
        raise ValueError("The capstone requires exactly five embedded queries")
    # Prevent benchmarking an IVF index from a previous embedding run.
    vectors = np.load("data/doc_vectors.npy")
    if not np.array_equal(load_index().vectors, vectors):
        raise ValueError("Stale IVF index: rerun python ivf.py")
    readiness = wait_for_indexes(client, expected_count=len(vectors))
    if args.query is not None:
        if not 1 <= args.query <= len(query_texts):
            parser.error("--query must be between 1 and 5")
        i = args.query - 1
        query_texts, query_vectors = query_texts[i:i+1], query_vectors[i:i+1]
        RESULTS_DIR = Path("results/demo")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: version(p) for p in ["qdrant-client", "numpy", "scikit-learn", "sentence-transformers"]},
        "qdrant_version": client.info().version,
        "warmups": WARMUPS, "repeats": REPEATS,
        "collections": readiness,
        "timing": "Qdrant includes HTTP/client overhead; IVF is in-process. Payload disabled.",
    }
    (RESULTS_DIR / "environment.json").write_text(json.dumps(environment, indent=2))
    run_distance_metric_comparison(client, query_texts, query_vectors, metadata)
    hnsw_results = run_hnsw_comparison(client, query_texts, query_vectors)
    exact_ids_by_query = [row["exact"]["ids"] for row in hnsw_results]
    run_ivf_comparison(query_texts, query_vectors, exact_ids_by_query)
    if args.query is None:
        write_report(RESULTS_DIR, Path("comparison.md"))
        print("Saved measured report -> comparison.md. Read it before submission.")
    else:
        print("Demo outputs saved separately in results/demo; full report preserved.")


if __name__ == "__main__":
    main()
