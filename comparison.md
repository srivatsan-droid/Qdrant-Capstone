# Comparison: Search Algorithms in Qdrant

Generated from the actual five-query run. Read and explain these observations before submission.

## Distance metrics

Query: **a question about a graphics card driver**

| Metric | Top-five IDs in rank order | Scores | Document norms |
|---|---|---|---|
| newsgroups_cosine | 5042, 2038, 2940, 2213, 1142 | 0.46341, 0.46021, 0.43471, 0.43298, 0.38822 | 2.72294, 2.10498, 2.64872, 2.25558, 2.14548 |
| newsgroups_euclidean | 5042, 2038, 2940, 2213, 5540 | 6.24993, 6.34016, 6.34154, 6.38306, 6.49511 | 2.72294, 2.10498, 2.64872, 2.25558, 2.31756 |
| newsgroups_dot | 1014, 2351, 5291, 2548, 1702 | 12.73417, 12.51997, 12.01287, 9.96271, 9.48217 | 6.04886, 5.76353, 5.37816, 4.44383, 6.12828 |

Cosine selected document 5042, while dot selected 1014. Their norms are 2.72294 and 6.04886. Dot multiplies angular similarity by document magnitude (and a fixed query magnitude), explaining how a different document can win.


Cosine measures direction; dot product measures direction and magnitude; Euclidean measures straight-line distance (lower score is closer). Cosine and dot scores are ranked highest first. All three collections receive identical raw input vectors, and Part 2 uses exact search to isolate the metric. Qdrant normalizes its cosine vectors internally. The model's final Normalize module is removed before embedding; no artificial scaling is added. Norm does not by itself establish document length, quality or relevance. Unit vectors would make these metrics mathematically equivalent in ranking, aside from ties and numerical effects.

## Combined results (Parts 3–5)

Default means default graph parameters with explicit benchmark thresholds. Under-tuned uses m=2 and ef_construct=8. Both set full_scan_threshold=0 and indexing_threshold=1 KB; indexing readiness is checked before timing.

| Method | Setting | Average top-five overlap | Average latency (ms) |
|---|---|---|---|
| Exact cosine | exhaustive | 100.00% | 7.7649 |
| hnsw_default | ef=16 | 100.00% | 4.2131 |
| hnsw_default | ef=64 | 100.00% | 4.5170 |
| hnsw_default | ef=128 | 100.00% | 4.7694 |
| hnsw_undertuned | ef=16 | 32.00% | 4.1176 |
| hnsw_undertuned | ef=64 | 44.00% | 4.2987 |
| hnsw_undertuned | ef=128 | 52.00% | 5.0149 |
| IVF | nprobe=1 | 92.00% | 0.0529 |
| IVF | nprobe=8 | 96.00% | 0.4898 |

Overlap is the size of the intersection with exact cosine top-five divided by five. It does not measure ordering or human relevance. Each query is warmed three times; latency is the mean of repeated measurements, then averaged across five queries. See results/environment.json for repetition count, software versions and index state. Result IDs and scores are from the final repetition. Tied scores can produce different valid top-five sets.

## Measured speed/accuracy tradeoff

hnsw_default: overlap stayed unchanged from 100.0% (ef=16) to 100.0% (ef=128); mean latency changed from 4.2131 to 4.7694 ms. hnsw_undertuned: overlap increased from 32.0% (ef=16) to 52.0% (ef=128); mean latency changed from 4.1176 to 5.0149 ms. IVF: overlap increased from 92.0% (nprobe=1) to 96.0% (nprobe=8); mean latency changed from 0.0529 to 0.4898 ms. These measurements describe this dataset and these five queries, rather than a universal winner.

Increasing HNSW ef allows the graph traversal to retain a larger candidate set. This generally improves the chance of finding the exact nearest neighbors, but costs additional distance evaluations. The under-tuned graph has fewer links and less construction effort, so the same search budget can produce different recall. Higher ef can help, but it does not rebuild missing graph links. Unchanged overlap may indicate that the tested queries already reached the available recall ceiling.

Increasing IVF nprobe searches more clusters. Documents excluded at a cluster boundary can enter the candidate set, so the exact cosine ranking inside that larger set can recover missing neighbors. Probing every cluster scans all vectors and matches exact cosine search apart from ties and floating-point effects. This implementation uses ordinary KMeans on normalized vectors and cosine centroid routing, an explicit approximation rather than spherical KMeans.

Latency need not increase monotonically in a small experiment because caching, scheduling and transport noise can mask computation costs. Qdrant timings include HTTP and client processing, whereas IVF timings cover an in-process function. They therefore cannot establish which indexing algorithm is intrinsically faster. Repeated warm measurements reduce noise but do not remove that measurement-boundary difference. A stronger performance study would use more queries, matched serving boundaries, and measured memory and build costs.

## Metric choice and index choice

Part 2 changes the definition of similarity while holding input data and exact-search procedure fixed. Parts 3–4 hold cosine similarity fixed while changing retrieval strategy. Metric choice changes the correct ranking; approximation can miss members of that ranking.

## Review questions

- Low nprobe misses vectors in unsearched clusters. Increasing nprobe expands candidates; nprobe=40 visits all 40 clusters. Here k in search_ivf means result count, not cluster count.
- Dot equals ||q|| × ||d|| × cosine. For a fixed query, document norms can change the ranking.
- One construction change: increase ef_construct and rebuild, holding m fixed. This spends more effort selecting graph neighbors. Increasing search-time ef is a separate way to spend more query work; neither guarantees perfect recall.
- IVF can suit large datasets and memory-sensitive systems, particularly with compressed vectors or optimized batched/GPU scans. Plain IVF still stores the vectors and requires centroid training; build/update speed is workload-dependent. This experiment did not measure memory or ingestion, so it cannot establish a production winner. Distribution changes can require retraining.

## Submission checks outside this benchmark

Confirm the local dashboard loads and record that observation in your README. Commit the measured results and this report in your own repository with meaningful commits. The five-query experiment is small; do not generalize its latency rankings to production.
