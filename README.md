# Qdrant Search Algorithms Capstone

Compares distance metrics and indexing strategies (Qdrant's HNSW vs. a
from-scratch IVF-style index) on a real Qdrant collection built from the
20 Newsgroups dataset.

## Setup

```bash
# 1. Start Qdrant (one command, persists data in ./qdrant_storage)
docker compose up -d
# confirm it's up: open http://localhost:6333/dashboard

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the pipeline, in order
python data.py            # sample ~6,000 docs from 20 Newsgroups
python embed.py           # embed docs + the 5 queries in queries.md
python qdrant_setup.py    # create the distance-metric and HNSW collections
python ivf.py             # build the from-scratch IVF index
python compare.py         # run every method against every query, save results/
```

`compare.py` is also the live demo: it hits all three distance metrics, both
HNSW configs at every `ef`, and both IVF `nprobe` values, printing results as
it goes, then writes `results/distance_metrics.json`, `results/hnsw.json`,
and `results/ivf.json`.

After running it, fill in `comparison.md` with the real numbers — it's
already structured with the tables and questions to answer, marked `TODO`.

## Project layout

```
qdrant-capstone/
├── docker-compose.yml      # starts Qdrant locally
├── requirements.txt
├── queries.md               # 5 test queries
├── data.py                  # loads/samples 20 Newsgroups
├── embed.py                 # embeds docs + queries with sentence-transformers
├── qdrant_setup.py          # creates the Qdrant collections
├── ivf.py                   # from-scratch IVF-style index (KMeans + nprobe search)
├── compare.py                # runs every method against the test queries
├── comparison.md             # written comparison and findings
└── results/                  # saved score/latency tables (JSON)
```

## Notes

- Model used: `all-MiniLM-L6-v2` (384-dim, sentence-transformers) — downloads
  once from Hugging Face and is cached locally after that.
- Re-running `qdrant_setup.py` recreates (drops + rebuilds) all five
  collections, so it's safe to re-run after changing embeddings.
