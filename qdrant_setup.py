"""
qdrant_setup.py

Creates every Qdrant collection the project needs:
  - Part 2: three collections with identical vectors/payload, one per
    distance metric (Cosine, Euclidean, Dot).
  - Part 3: two Cosine collections with different hnsw_config — a default
    one, and a deliberately under-tuned one (low m, low ef_construct).
"""

import pickle
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    OptimizersConfigDiff,
    VectorParams,
)

QDRANT_URL = "http://localhost:6333"

DOC_VECTORS_PATH = Path("data/doc_vectors.npy")
DOC_METADATA_PATH = Path("data/doc_metadata.pkl")

DISTANCE_COLLECTIONS = {
    "newsgroups_cosine": Distance.COSINE,
    "newsgroups_euclidean": Distance.EUCLID,
    "newsgroups_dot": Distance.DOT,
}

HNSW_DEFAULT_COLLECTION = "newsgroups_hnsw_default"
HNSW_UNDERTUNED_COLLECTION = "newsgroups_hnsw_undertuned"

# Deliberately weak: few graph edges per node (m) and little effort spent
# building the graph (ef_construct) -> should recall worse than the default.
UNDERTUNED_HNSW_CONFIG = HnswConfigDiff(m=2, ef_construct=8, full_scan_threshold=10)


def recreate_collection(client, collection_name, **kwargs):
    """Destructively rebuild this project's collection; keep the API explicit."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(collection_name=collection_name, **kwargs)


def wait_for_indexes(client, expected_count=None, timeout=300):
    """Require complete indexing before allowing the HNSW experiment.

    Upload acknowledgement does not imply background graph construction is done.
    Counts are approximate during optimization; wait for a stable green state.
    """
    deadline = time.monotonic() + timeout
    snapshots = {}
    names = [HNSW_DEFAULT_COLLECTION, HNSW_UNDERTUNED_COLLECTION]
    while True:
        ready = True
        for name in names:
            info = client.get_collection(name)
            count = client.count(name, exact=True).count
            config = info.config.hnsw_config
            if config.full_scan_threshold != 10:
                raise RuntimeError(f"{name}: rerun qdrant_setup.py to apply benchmark thresholds")
            status = getattr(info.status, "value", info.status)
            optimizer_status = getattr(info.optimizer_status, "value", info.optimizer_status)
            snapshots[name] = {
                "points_count": count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(status),
                "optimizer_status": str(optimizer_status),
                "m": config.m, "ef_construct": config.ef_construct,
                "full_scan_threshold": config.full_scan_threshold,
                "indexing_threshold": info.config.optimizer_config.indexing_threshold,
            }
            ready &= (count > 0 and (expected_count is None or count == expected_count)
                      and info.indexed_vectors_count == count and str(status) == "green"
                      and str(optimizer_status).lower() == "ok")
        if ready:
            return snapshots
        if time.monotonic() >= deadline:
            raise TimeoutError(f"HNSW indexing not ready: {snapshots}")
        time.sleep(1)


def load_vectors_and_metadata():
    if not DOC_VECTORS_PATH.exists():
        raise FileNotFoundError(f"{DOC_VECTORS_PATH} not found — run `python embed.py` first.")
    vectors = np.load(DOC_VECTORS_PATH)
    with open(DOC_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    if len(vectors) != len(metadata) or not np.isfinite(vectors).all():
        raise ValueError("Vectors and metadata must be aligned and finite")
    return vectors, metadata


def build_points(vectors, metadata):
    points = []
    for idx, (vector, meta) in enumerate(zip(vectors, metadata)):
        points.append(
            PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={"category": meta["category"], "text": meta["text"]},
            )
        )
    return points


def upsert_in_batches(client, collection_name, points, batch_size=256):
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i + batch_size])


def create_distance_collections(client, vectors, points):
    dim = vectors.shape[1]
    for name, distance in DISTANCE_COLLECTIONS.items():
        print(f"Creating '{name}' (distance={distance})...")
        recreate_collection(client,
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=distance),
        )
        upsert_in_batches(client, name, points)
        print(f"  Upserted {len(points)} points into '{name}'.")


def create_hnsw_collections(client, vectors, points):
    dim = vectors.shape[1]

    print(f"Creating '{HNSW_DEFAULT_COLLECTION}' (default graph, benchmark thresholds)...")
    recreate_collection(client,
        collection_name=HNSW_DEFAULT_COLLECTION,
        hnsw_config=HnswConfigDiff(full_scan_threshold=10),
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=1, default_segment_number=1),
    )
    upsert_in_batches(client, HNSW_DEFAULT_COLLECTION, points)

    print(f"Creating '{HNSW_UNDERTUNED_COLLECTION}' (m=2, ef_construct=8)...")
    recreate_collection(client,
        collection_name=HNSW_UNDERTUNED_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=1, default_segment_number=1),
        hnsw_config=UNDERTUNED_HNSW_CONFIG,
    )
    upsert_in_batches(client, HNSW_UNDERTUNED_COLLECTION, points)

    print("Waiting for HNSW indexes...")
    print(wait_for_indexes(client, expected_count=len(points)))


def main():
    vectors, metadata = load_vectors_and_metadata()
    points = build_points(vectors, metadata)

    client = QdrantClient(url=QDRANT_URL)

    create_distance_collections(client, vectors, points)
    create_hnsw_collections(client, vectors, points)

    print("\nAll collections created. Check http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
