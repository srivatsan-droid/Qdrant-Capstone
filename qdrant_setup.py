"""
qdrant_setup.py

Creates every Qdrant collection the project needs:
  - Part 2: three collections with identical vectors/payload, one per
    distance metric (Cosine, Euclidean, Dot).
  - Part 3: two Cosine collections with different hnsw_config — a default
    one, and a deliberately under-tuned one (low m, low ef_construct).
"""

import pickle
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
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
UNDERTUNED_HNSW_CONFIG = HnswConfigDiff(m=2, ef_construct=8)


def load_vectors_and_metadata():
    if not DOC_VECTORS_PATH.exists():
        raise FileNotFoundError(f"{DOC_VECTORS_PATH} not found — run `python embed.py` first.")
    vectors = np.load(DOC_VECTORS_PATH)
    with open(DOC_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
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
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=distance),
        )
        upsert_in_batches(client, name, points)
        print(f"  Upserted {len(points)} points into '{name}'.")


def create_hnsw_collections(client, vectors, points):
    dim = vectors.shape[1]

    print(f"Creating '{HNSW_DEFAULT_COLLECTION}' (default HNSW config)...")
    client.recreate_collection(
        collection_name=HNSW_DEFAULT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    upsert_in_batches(client, HNSW_DEFAULT_COLLECTION, points)

    print(f"Creating '{HNSW_UNDERTUNED_COLLECTION}' (m=4, ef_construct=16)...")
    client.recreate_collection(
        collection_name=HNSW_UNDERTUNED_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        hnsw_config=UNDERTUNED_HNSW_CONFIG,
    )
    upsert_in_batches(client, HNSW_UNDERTUNED_COLLECTION, points)

    print("Done creating HNSW comparison collections.")


def main():
    vectors, metadata = load_vectors_and_metadata()
    points = build_points(vectors, metadata)

    client = QdrantClient(url=QDRANT_URL)

    create_distance_collections(client, vectors, points)
    create_hnsw_collections(client, vectors, points)

    print("\nAll collections created. Check http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
