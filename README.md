# Complaint Intelligence System

An NLP pipeline that processes 200K consumer complaints from the CFPB database, comparing older and newer techniques for embedding, clustering, and retrieval.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://complaint-intelligence-system.streamlit.app/)
&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Interactive Dashboard:** [complaint-intelligence-system.streamlit.app](https://complaint-intelligence-system.streamlit.app/)

![Overview](screenshots/overview.png)

## What it does

Takes raw complaint text, cleans it, generates embeddings with two different models, clusters the complaints using two different methods, and provides a semantic search interface. Everything is benchmarked so you can see the actual tradeoffs.

The pipeline compares:
- **Embeddings**: MiniLM (384d, fast) vs BGE (768d, more accurate)
- **Clustering**: KMeans (fixed k) vs BERTopic (auto-discovers topics)
- **Retrieval**: Vector search, BM25, hybrid, and reranked hybrid

## System Architecture

```mermaid
graph TD
    A[CFPB Raw Data] -->|run_pipeline.py| B[Text Cleaning & Preprocessing]
    
    subgraph Clustering Subsystem
        B --> E1[KMeans Clustering]
        B --> E2[BERTopic Pipeline]
        E2 --> E2a[UMAP Dimension Reduction]
        E2a --> E2b[HDBSCAN Density Clustering]
    end

    subgraph Embedding Subsystem
        B --> C1[all-MiniLM-L6-v2]
        B --> C2[bge-base-en-v1.5]
    end
    
    C1 & C2 --> G[FAISS Dense Index]
    B --> H[BM25 Sparse Index]
    
    subgraph Retrieval Pipeline
        G & H --> I[Hybrid Search]
        I -->|Reciprocal Rank Fusion| J[Candidate Generation]
        J -->|Cross-Encoder Reranker| K[Reranked Top Results]
    end
    
    K --> L[LLM Summarizer / RAG Context]
    L --> M[Streamlit Web Dashboard]
```

## Key Engineering Decisions & Insights

### 1. Sparse vs. Dense Retrieval (BM25 vs. FAISS)
* **FAISS (Dense)** provides conceptual matching (e.g., searching for *"stolen card"* retrieves complaints mentioning *"unauthorized charge"* or *"lost wallet"*), and runs under **35ms**. However, it suffers from keyword mismatch.
* **BM25 (Sparse)** guarantees matches for specific product terms, account numbers, or unique company names but misses conceptual synonyms and suffers from Python search overhead (~580ms).
* **Hybrid Reranked Retrieval** solves this by using BM25 and FAISS for fast candidate generation, then ranking results with `MS-MARCO-MiniLM-L-6-v2`. This delivers the highest precision results at the cost of ~300ms additional latency.

### 2. Embedding Trade-offs: Throughput vs. Semantic Quality
* **MiniLM** (384-dimensions) achieves high throughput (**374.6 texts/sec**), making it highly scalable for real-time applications.
* **BGE** (768-dimensions) has a lower throughput (**59.7 texts/sec**) but achieves significantly higher intra-cluster coherence (**0.77** vs. **0.53**).
* **Alignment Analysis**: The two embedding spaces only share **38.5% of their top-10 nearest neighbors** for any given complaint. This confirms that higher-dimensional models capture structurally different semantic relationships than smaller ones.

### 3. Clustering Mechanics: Forced Partitioning vs. Density-Based
* **KMeans** forces every complaint into a group, which guarantees coverage but creates high intra-cluster variance on messy, real-world data.
* **BERTopic** leverages HDBSCAN to isolate only dense, high-confidence topic regions, filtering out **55%** of the complaints as outliers/noise. This honest approach produces highly distinct clusters for the remaining data, which we auto-labeled using Gemini.

## Results

Ran on 200K complaints using a T4 GPU on Google Colab.

### Embeddings

| Model | Dim | Speed | Cosine Sim (mean) | Intra-cluster Coherence |
|---|---|---|---|---|
| MiniLM | 384 | 374.6 texts/sec | 0.41 | 0.53 |
| BGE | 768 | 59.7 texts/sec | 0.72 | 0.77 |

BGE is 6x slower but produces much tighter clusters. The two models only agree on 38.5% of top-10 neighbors, meaning they capture different aspects of the text.

### Clustering

| Metric | KMeans (k=6) | BERTopic |
|---|---|---|
| Clusters found | 6 | 30 |
| Outliers | 0 | 110,456 (55%) |
| Silhouette | 0.0338 | 0.0301 |

KMeans forces every complaint into a cluster. BERTopic flags 55% as noise, which is honest but means over half the data gets no label. Both have low silhouette scores, which makes sense since complaint text is messy and overlapping.

### Retrieval latency

| Method | p50 (ms) | p95 (ms) |
|---|---|---|
| Vector (FAISS) | 35 | 41 |
| BM25 | 589 | 929 |
| Hybrid (RRF) | 614 | 959 |
| Reranked Hybrid | 911 | 1,356 |

Pure vector search is fast. Adding BM25 and reranking improves result quality but adds significant latency. Whether that tradeoff is worth it depends on the use case.

## How to run

### Setup

```bash
git clone https://github.com/AswaniSahoo/complaint-intelligence-system.git
cd complaint-intelligence-system

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Get the data

Download from the [CFPB website](https://www.consumerfinance.gov/data-research/consumer-complaints/) and put it in `data/raw/complaints.csv`.

### Run the pipeline

```bash
# Full run with both models and benchmarks
python run_pipeline.py --sample-size 200000 --model both --clustering both --benchmark

# Quick run (MiniLM + KMeans only)
python run_pipeline.py

# With LLM summarization (needs GEMINI_API_KEY in .env)
python run_pipeline.py --with-llm
```

### Launch the dashboard

```bash
streamlit run app/app.py
```

### Running on Colab

For large runs, use Colab with a T4 GPU. The full pipeline notebook is at `notebooks/Full_pipeline_output.ipynb`.

```python
!git clone https://github.com/AswaniSahoo/complaint-intelligence-system.git
%cd complaint-intelligence-system
!pip install -r requirements.txt -q

!mkdir -p data/raw
!wget -q -O data/raw/complaints.csv.zip \
    "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
!cd data/raw && unzip -o complaints.csv.zip && rm complaints.csv.zip

!python run_pipeline.py --sample-size 200000 --model both --clustering both --benchmark
```

## Pipeline flags

| Flag | What it does | Default |
|---|---|---|
| `--sample-size N` | How many complaints to process | 15000 |
| `--model {minilm,bge,both}` | Which embedding model(s) | minilm |
| `--clustering {kmeans,bertopic,both}` | Which clustering method(s) | kmeans |
| `--benchmark` | Run retrieval latency tests | off |
| `--with-llm` | Generate LLM summaries | off |
| `--provider {gemini,groq,together}` | LLM provider | gemini |

## Dashboard

7 pages:

- **Overview** - complaint counts, product/issue distributions, time trends
- **Clusters** - drill into each cluster, see what products/issues it contains
- **Complaint Viewer** - browse and filter individual complaints
- **Semantic Search** - search complaints by meaning using FAISS
- **Embedding Comparison** - MiniLM vs BGE metrics side by side
- **Clustering Comparison** - KMeans vs BERTopic quality metrics
- **Retrieval Benchmark** - latency comparison across all four retrievers

<details>
<summary>Screenshots</summary>

### Cluster Analysis
![Clusters](screenshots/clusters.png)

### Complaint Viewer
![Viewer](screenshots/viewer.png)

### Semantic Search
![Semantic Search](screenshots/semantic_search.png)

### Embedding Comparison
![Embedding Comparison](screenshots/embedding_comparison.png)

### Clustering Comparison
![Clustering Comparison](screenshots/clustering_comparison.png)

### Retrieval Benchmark
![Retrieval Benchmark](screenshots/retrieval_benchmark.png)

</details>

## Project structure

```
├── app/
│   └── app.py                     # Streamlit dashboard
├── src/
│   ├── preprocess.py              # Text cleaning
│   ├── embeddings.py              # Embedding generation (MiniLM + BGE)
│   ├── embedding_benchmark.py     # Embedding comparison metrics
│   ├── clustering.py              # KMeans + BERTopic
│   ├── topic_labeler.py           # LLM-based topic labeling
│   ├── rag.py                     # FAISS-based search
│   ├── llm_utils.py               # LLM provider abstraction
│   ├── visualizer.py              # UMAP projections
│   ├── retrievers/
│   │   ├── base.py                # Retriever interface
│   │   ├── vector_retriever.py    # FAISS search
│   │   ├── bm25_retriever.py      # BM25 keyword search
│   │   ├── hybrid_retriever.py    # Vector + BM25 fusion
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   └── reranked_retriever.py  # Two-stage retrieval
│   └── evaluation/
│       └── retrieval_benchmark.py # Latency benchmarks
├── data/
│   ├── raw/                       # CFPB data (not committed)
│   ├── processed/                 # Cleaned data + embeddings
│   └── results/                   # Benchmark outputs (JSON)
├── notebooks/
│   └── Full_pipeline_output.ipynb # Complete Colab run
├── tests/
├── run_pipeline.py                # Main pipeline script
└── requirements.txt
```

## Tech stack

- `sentence-transformers` for embeddings
- `bertopic`, `hdbscan`, `umap-learn` for topic modeling
- `faiss-cpu` for vector search
- `rank-bm25` for keyword search
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- `streamlit` + `plotly` for the dashboard
- Google Gemini for topic labeling (optional)

## License

MIT
