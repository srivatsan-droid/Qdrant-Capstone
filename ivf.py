"""
ivf.py

A minimal IVF-style ("inverted file") index, built from scratch, since
Qdrant only ships HNSW as a selectable dense-vector index. Vectors are
clustered with KMeans; a search first finds the nprobe closest cluster
centroids to the query, then does exact cosine similarity only over the
vectors assigned to those clusters.
"""

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

DOC_VECTORS_PATH = Path("data/doc_vectors.npy")
IVF_INDEX_PATH = Path("data/ivf_index.pkl")

DEFAULT_N_CLUSTERS = 40


@dataclass
class IVFIndex:
    centroids: np.ndarray                                # (k, dim)
    cluster_to_ids: dict                                  # cluster_id -> list[int]
    vectors: np.ndarray                                   # (n, dim), original order
    normalized_vectors: np.ndarray = field(repr=False)     # unit vectors, for cosine search


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return matrix / norms


def build_ivf_index(vectors: np.ndarray, k: int = DEFAULT_N_CLUSTERS, seed: int = 42) -> IVFIndex:
    """Cluster vectors into k clusters with KMeans and record which vector
    ids fall into each cluster, along with the cluster centroids."""
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    assignments = kmeans.fit_predict(vectors)

    cluster_to_ids = {i: [] for i in range(k)}
    for vector_id, cluster_id in enumerate(assignments):
        cluster_to_ids[int(cluster_id)].append(vector_id)

    return IVFIndex(
        centroids=kmeans.cluster_centers_,
        cluster_to_ids=cluster_to_ids,
        vectors=vectors,
        normalized_vectors=_normalize(vectors),
    )


def search_ivf(query_vector: np.ndarray, index: IVFIndex, nprobe: int, k: int):
    """Find the nprobe nearest centroids to the query (cosine similarity),
    then compute exact cosine similarity only against vectors belonging to
    those clusters. Returns a list of (vector_id, score) tuples, top-k."""
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)

    # Step 1: rank clusters by how similar their centroid is to the query.
    normalized_centroids = _normalize(index.centroids)
    centroid_scores = normalized_centroids @ query_norm
    nearest_clusters = np.argsort(-centroid_scores)[:nprobe]

    # Step 2: gather candidate vector ids from just those clusters.
    candidate_ids = []
    for cluster_id in nearest_clusters:
        candidate_ids.extend(index.cluster_to_ids[int(cluster_id)])

    if not candidate_ids:
        return []

    # Step 3: exact cosine similarity search, restricted to the candidates.
    candidate_ids = np.array(candidate_ids)
    candidate_vectors = index.normalized_vectors[candidate_ids]
    scores = candidate_vectors @ query_norm

    top_k_local = np.argsort(-scores)[:k]
    return [(int(candidate_ids[i]), float(scores[i])) for i in top_k_local]


def save_index(index: IVFIndex, path: Path = IVF_INDEX_PATH):
    with open(path, "wb") as f:
        pickle.dump(index, f)


class _ModuleRemapUnpickler(pickle.Unpickler):
    """Unpickler that treats classes recorded as living in '__main__' as
    living in this module instead.

    Pickle stores a class reference as (module_name, class_name), using
    whatever module was executing at save time. If the index was built by
    running `python ivf.py` directly, IVFIndex gets recorded under
    '__main__' rather than 'ivf'. Loading it from any other entry point
    (e.g. compare.py, where compare.py is __main__) then fails with
    AttributeError, even though the class is right here. This remaps that
    one case so the pickle loads correctly regardless of how it was built.
    """

    def find_class(self, module, name):
        if module == "__main__" and name == "IVFIndex":
            module = __name__
        return super().find_class(module, name)


def load_index(path: Path = IVF_INDEX_PATH) -> IVFIndex:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python ivf.py` first.")
    with open(path, "rb") as f:
        return _ModuleRemapUnpickler(f).load()


def main():
    if not DOC_VECTORS_PATH.exists():
        raise FileNotFoundError(f"{DOC_VECTORS_PATH} not found — run `python embed.py` first.")

    vectors = np.load(DOC_VECTORS_PATH)
    print(f"Clustering {vectors.shape[0]} vectors into {DEFAULT_N_CLUSTERS} clusters...")

    start = time.perf_counter()
    index = build_ivf_index(vectors, k=DEFAULT_N_CLUSTERS)
    elapsed = time.perf_counter() - start
    print(f"Built IVF index in {elapsed:.2f}s.")

    sizes = [len(ids) for ids in index.cluster_to_ids.values()]
    print(f"Cluster sizes -> min={min(sizes)}, max={max(sizes)}, "
          f"avg={sum(sizes) / len(sizes):.1f}")

    Path("data").mkdir(exist_ok=True)
    save_index(index)
    print(f"Saved index -> {IVF_INDEX_PATH}")


if __name__ == "__main__":
    main()