"""Generate comparison.md from this run's real measurements, never example values."""
import json
from pathlib import Path
from statistics import mean

import numpy as np


def write_report(results_dir=Path('results'), destination=Path('comparison.md')):
    metrics, hnsw, ivf = [json.loads((results_dir / name).read_text()) for name in
                         ['distance_metrics.json', 'hnsw.json', 'ivf.json']]
    if not (len(metrics) == len(hnsw) == len(ivf) == 5):
        raise ValueError('A submission report requires all five queries')
    if not ([r['query'] for r in metrics] == [r['query'] for r in hnsw] == [r['query'] for r in ivf]):
        raise ValueError('Result files contain different query sets')
    rows = [('Exact cosine', 'exhaustive', 1.0, mean(r['exact']['latency_ms'] for r in hnsw))]
    for label in ['hnsw_default', 'hnsw_undertuned']:
        for ef in sorted(hnsw[0][label], key=int):
            rows.append((label, 'ef=' + ef,
                         mean(r[label][ef]['overlap_with_exact'] for r in hnsw),
                         mean(r[label][ef]['latency_ms'] for r in hnsw)))
    for nprobe in sorted(ivf[0]['results'], key=int):
        rows.append(('IVF', 'nprobe=' + nprobe,
                     mean(r['results'][nprobe]['overlap_with_exact'] for r in ivf),
                     mean(r['results'][nprobe]['latency_ms'] for r in ivf)))
    table = '\n'.join(f'| {name} | {setting} | {ov:.2%} | {ms:.4f} |' for name, setting, ov, ms in rows)
    vectors = np.load('data/doc_vectors.npy')
    norms = np.linalg.norm(vectors, axis=1)
    differing = next((r for r in metrics if len({tuple(x['id'] for x in hits)
                     for hits in r['results'].values()}) > 1), None)
    metric_text = 'No ranking difference was observed. Do not claim one: inspect vector norms and try a new query, then rerun embed.py, qdrant_setup.py, ivf.py and compare.py. Part 2 remains incomplete until a measured differing case is documented.'
    if differing:
        metric_text = f"Query: **{differing['query']}**\n\n| Metric | Top-five IDs in rank order | Scores | Document norms |\n|---|---|---|---|\n"
        for name, hits in differing['results'].items():
            metric_text += f"| {name} | {', '.join(str(h['id']) for h in hits)} | {', '.join(format(h['score'], '.5f') for h in hits)} | {', '.join(format(norms[h['id']], '.5f') for h in hits)} |\n"
        cosine = differing['results']['newsgroups_cosine']
        dot = differing['results']['newsgroups_dot']
        if cosine[0]['id'] != dot[0]['id']:
            a, b = cosine[0]['id'], dot[0]['id']
            metric_text += f'\nCosine selected document {a}, while dot selected {b}. Their norms are {norms[a]:.5f} and {norms[b]:.5f}. Dot multiplies angular similarity by document magnitude (and a fixed query magnitude), explaining how a different document can win.\n'
        else:
            metric_text += '\nThe table identifies the measured ordering difference; cosine and dot may still agree on the first result. Euclidean also depends on vector norms, through squared distance = ||q||² + ||d||² − 2(q·d).\n'
    changes = []
    for name in ['hnsw_default', 'hnsw_undertuned', 'IVF']:
        selected = [r for r in rows if r[0] == name]
        first, last = selected[0], selected[-1]
        direction = 'increased' if last[2] > first[2] else 'decreased' if last[2] < first[2] else 'stayed unchanged'
        changes.append(f'{name}: overlap {direction} from {first[2]:.1%} ({first[1]}) to {last[2]:.1%} ({last[1]}); mean latency changed from {first[3]:.4f} to {last[3]:.4f} ms.')
    analysis = ' '.join(changes) + ''' These measurements describe this dataset and these five queries, rather than a universal winner.

Increasing HNSW ef allows the graph traversal to retain a larger candidate set. This generally improves the chance of finding the exact nearest neighbors, but costs additional distance evaluations. The under-tuned graph has fewer links and less construction effort, so the same search budget can produce different recall. Higher ef can help, but it does not rebuild missing graph links. Unchanged overlap may indicate that the tested queries already reached the available recall ceiling.

Increasing IVF nprobe searches more clusters. Documents excluded at a cluster boundary can enter the candidate set, so the exact cosine ranking inside that larger set can recover missing neighbors. Probing every cluster scans all vectors and matches exact cosine search apart from ties and floating-point effects. This implementation uses ordinary KMeans on normalized vectors and cosine centroid routing, an explicit approximation rather than spherical KMeans.

Latency need not increase monotonically in a small experiment because caching, scheduling and transport noise can mask computation costs. Qdrant timings include HTTP and client processing, whereas IVF timings cover an in-process function. They therefore cannot establish which indexing algorithm is intrinsically faster. Repeated warm measurements reduce noise but do not remove that measurement-boundary difference. A stronger performance study would use more queries, matched serving boundaries, and measured memory and build costs.'''
    output = f'''# Comparison: Search Algorithms in Qdrant

Generated from the actual five-query run. Read and explain these observations before submission.

## Distance metrics

{metric_text}

Cosine measures direction; dot product measures direction and magnitude; Euclidean measures straight-line distance (lower score is closer). Cosine and dot scores are ranked highest first. All three collections receive identical raw input vectors, and Part 2 uses exact search to isolate the metric. Qdrant normalizes its cosine vectors internally. The model's final Normalize module is removed before embedding; no artificial scaling is added. Norm does not by itself establish document length, quality or relevance. Unit vectors would make these metrics mathematically equivalent in ranking, aside from ties and numerical effects.

## Combined results (Parts 3–5)

Default means default graph parameters with explicit benchmark thresholds. Under-tuned uses m=2 and ef_construct=8. Both set full_scan_threshold=10 and indexing_threshold=1 KB; indexing readiness is checked before timing.

| Method | Setting | Average top-five overlap | Average latency (ms) |
|---|---|---|---|
{table}

Overlap is the size of the intersection with exact cosine top-five divided by five. It does not measure ordering or human relevance. Each query is warmed three times; latency is the mean of repeated measurements, then averaged across five queries. See results/environment.json for repetition count, software versions and index state. Result IDs and scores are from the final repetition. Tied scores can produce different valid top-five sets.

## Measured speed/accuracy tradeoff

{analysis}

## Metric choice and index choice

Part 2 changes the definition of similarity while holding input data and exact-search procedure fixed. Parts 3–4 hold cosine similarity fixed while changing retrieval strategy. Metric choice changes the correct ranking; approximation can miss members of that ranking.

## Review questions

- Low nprobe misses vectors in unsearched clusters. Increasing nprobe expands candidates; nprobe=40 visits all 40 clusters. Here k in search_ivf means result count, not cluster count.
- Dot equals ||q|| × ||d|| × cosine. For a fixed query, document norms can change the ranking.
- One construction change: increase ef_construct and rebuild, holding m fixed. This spends more effort selecting graph neighbors. Increasing search-time ef is a separate way to spend more query work; neither guarantees perfect recall.
- IVF can suit large datasets and memory-sensitive systems, particularly with compressed vectors or optimized batched/GPU scans. Plain IVF still stores the vectors and requires centroid training; build/update speed is workload-dependent. This experiment did not measure memory or ingestion, so it cannot establish a production winner. Distribution changes can require retraining.

## Submission checks outside this benchmark

Confirm the local dashboard loads and record that observation in your README. Commit the measured results and this report in your own repository with meaningful commits. The five-query experiment is small; do not generalize its latency rankings to production.
'''
    destination.write_text(output, encoding='utf-8')
    print(f'Analysis section: {len(analysis.split())} words')


if __name__ == '__main__':
    write_report()
