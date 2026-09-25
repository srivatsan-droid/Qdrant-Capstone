# Comparison: Search Algorithms in Qdrant

> **How to fill this in:** run the pipeline end-to-end (`data.py` → `embed.py`
> → `qdrant_setup.py` → `ivf.py` → `compare.py`) and then pull the actual
> numbers out of `results/distance_metrics.json`, `results/hnsw.json`, and
> `results/ivf.json` to replace every `TODO` below. The structure, the
> questions to answer, and what to look for are already laid out — you're
> filling in real numbers and your own explanation, not writing from
> scratch.
> **If you're re-running after the embed.py / qdrant_setup.py fix:** all
> three `results/*.json` files are now stale (they were generated from
> unit-normalized embeddings and a milder under-tuned HNSW config) and need
> to be regenerated from scratch — delete `data/*.pkl`, `data/*.npy`, and
> `data/ivf_index.pkl`, then re-run the full pipeline.
## Part 2: Distance metrics — where ranking order differs

`results/distance_metrics.json` has the top-5 for every query under Cosine,
Euclidean, and Dot product, on identical vectors and payload.

- **Query used:** TODO (pick the one from `queries.md` where the top result
  or ranking order visibly differs between at least two metrics)
- **Cosine top result:** TODO id / category / score
- **Euclidean top result:** TODO id / category / score
- **Dot product top result:** TODO id / category / score

**Why it differs:** Cosine similarity only looks at the *angle* between two
vectors — it ignores their length entirely, since both vectors get
normalized before comparison. Euclidean distance measures *straight-line*
distance in the embedding space, so two vectors can point in nearly the same
direction but still be "far apart" if one is much longer than the other.
Dot product combines both angle and magnitude — a document embedding with a
larger norm (often correlated with longer or more topically "dense" text)
can outscore a better-angled but shorter embedding purely because of its
length. TODO: name the specific document pair from your run and which of
these three explanations actually applies to it.
*(This only works because `embed.py` strips `all-MiniLM-L6-v2`'s built-in
Normalize module. With unit-normalized embeddings, dot product and cosine
give the exact same ranking for every query, and Euclidean distance is a
monotonic function of cosine similarity — so all three metrics would be
mathematically forced to agree, and there'd be nothing to find here.)*

## Part 3: Exact search vs. HNSW (default vs. under-tuned)

| Method | ef | Avg. top-5 overlap with exact | Avg. latency (ms) |
|---|---|---|---|
| Exact (brute-force) | — | 1.00 | TODO |
| HNSW default | 4 | TODO | TODO |
| HNSW default | 16 | TODO | TODO |
| HNSW default | 64 | TODO | TODO |
| HNSW default | 128 | TODO | TODO |
| HNSW under-tuned (m=2, ef_construct=8) | 4 | TODO | TODO |
| HNSW under-tuned (m=2, ef_construct=8) | 16 | TODO | TODO |
| HNSW under-tuned (m=2, ef_construct=8) | 64 | TODO | TODO |
| HNSW under-tuned (m=2, ef_construct=8) | 128 | TODO | TODO |

*(Average each column across all 5 queries from `results/hnsw.json`.)*

**What raising `ef` does:** TODO — describe, using your actual numbers,
whether overlap climbs toward 1.0 as `ef` increases, and what that costs in
latency. Tie it back to what `ef` controls: how many candidate nodes the
graph search keeps "in play" during traversal before returning the best
ones it found — a higher `ef` means more of the graph gets explored, closer
to exhaustive, at the cost of more distance computations per query.

**Under-tuned vs. default at the same `ef`:** TODO — does the under-tuned
collection ever catch up to the default collection's overlap at high `ef`?
Why or why not, given that `m` (edges per node) and `ef_construct` (search
effort during graph *construction*) were fixed low when the collection was
built and can't be fixed at search time.

## Part 4: From-scratch IVF index

| nprobe | Avg. top-5 overlap with exact | Avg. latency (ms) |
|---|---|---|
| 1 | TODO | TODO |
| 8 | TODO | TODO |

*(Average across all 5 queries from `results/ivf.json`.)*

**Why low `nprobe` disagrees with exact search:** at `nprobe=1`, `search_ivf`
only looks inside the single cluster whose centroid is closest to the query.
If the true nearest neighbors happen to sit near a cluster boundary, they
can be assigned to a *different* cluster than the one the query's centroid
search picks, so they're never examined at all — a classic case of a good
approximate index still missing "edge" points. Raising `nprobe` widens the
net to more clusters, shrinking that blind spot, until at
`nprobe = k` (all clusters) it becomes equivalent to exact search.

## Part 5: Combined table — every method, one view

| Method | Setting | Avg. top-5 overlap with exact | Avg. latency (ms) |
|---|---|---|---|
| Exact (brute-force) | — | 1.00 | TODO |
| HNSW default | ef=4 | TODO | TODO |
| HNSW default | ef=16 | TODO | TODO |
| HNSW default | ef=64 | TODO | TODO |
| HNSW default | ef=128 | TODO | TODO |
| HNSW under-tuned | ef=4 | TODO | TODO |
| HNSW under-tuned | ef=16 | TODO | TODO |
| HNSW under-tuned | ef=64 | TODO | TODO |
| HNSW under-tuned | ef=128 | TODO | TODO |
| IVF (from scratch) | nprobe=1 | TODO | TODO |
| IVF (from scratch) | nprobe=8 | TODO | TODO |

### Which method converges to exact search faster, and why

TODO (roughly 200–300 words). Points to hit:

- Compare how quickly HNSW's overlap approaches 1.0 as `ef` grows vs. how
  quickly IVF's overlap approaches 1.0 as `nprobe` grows, using your actual
  numbers.
- Both are the same underlying tradeoff from the indexing unit: spend more
  search-time compute (explore more graph nodes / probe more clusters) to
  trade speed for accuracy, approaching brute-force as the "more work"
  parameter approaches its maximum (`ef` → collection size, `nprobe` → `k`).
- HNSW's graph structure lets it jump between far-apart regions of the space
  in a few hops, so it can often recover high overlap at a fraction of
  exact search's latency. IVF's accuracy is coupled more directly to how
  many clusters you're willing to scan — say concretely, from your numbers,
  whether IVF needed a larger fraction of "total work" than HNSW did to
  reach comparable overlap.

### Distance metric vs. index choice — separate decisions

Referencing the Part 2 example above: the ranking changed between Cosine and
Dot product even though every method in Parts 3–4 (exact, HNSW, IVF) used
the *same* index and the *same* dataset. That's because distance metric is a
property of how similarity is *defined* — it determines what "closer" even
means — while HNSW/IVF/exact are strategies for *finding* the vectors that
are closest under whatever metric you picked. Changing the metric changes
the correct answer; changing the index only changes how fast (and how
reliably) you find that answer.

## Review-session prep notes

- **Why IVF disagrees with exact search at low nprobe, and what raising
  nprobe does:** see Part 4 above.
- **Why cosine vs. dot product can return a different top result from
  identical vectors:** see Part 2 above — it comes down to whether vector
  magnitude affects the score.
- **What single change would make the under-tuned HNSW collection behave
  more like the default one:** raising `m` and `ef_construct` at
  *collection-creation time* (not `ef` at search time) — a sparser graph
  built with a low-effort construction pass has structural gaps that no
  amount of search-time `ef` can fully route around.
- **When would a real system want IVF instead of HNSW:** IVF's index is
  cheap to build (just k-means) and cheap to update incrementally (append a
  vector to its nearest cluster's list) compared to inserting into an HNSW
  graph, and its memory footprint is smaller since it doesn't store a
  multi-layer graph — so IVF tends to win when the dataset is huge,
  writes/updates are frequent, or memory is the binding constraint, and a
  small accuracy hit at a given latency budget is acceptable. TODO: tie this
  back to any latency/memory pattern you actually observed in Part 4.
