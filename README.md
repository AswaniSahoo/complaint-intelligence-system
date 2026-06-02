# 🧠 Complaint Intelligence System

> A production-grade NLP pipeline for customer complaint analysis — comparing **old-school (2022)** vs **state-of-the-art (2024-26)** techniques across embeddings, clustering, retrieval, and evaluation.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-MiniLM%20%7C%20BGE-orange.svg)](https://sbert.net)
[![BERTopic](https://img.shields.io/badge/Clustering-BERTopic-green.svg)](https://maartengr.github.io/BERTopic)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-red.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io)

---

## Why This Project?

Most NLP portfolio projects stop at "I built a chatbot." This one goes deeper — it **benchmarks old methods against new ones** and shows you exactly what improved, by how much, and why.

| Component | Old (Baseline, 2022-era) | New (SOTA, 2024-26) | Improvement |
|---|---|---|---|
| **Embedding Model** | `all-MiniLM-L6-v2` (384d) | `BAAI/bge-base-en-v1.5` (768d) | Higher MTEB scores, better cluster separation |
| **Clustering** | KMeans (fixed k=6) | BERTopic (UMAP + HDBSCAN + c-TF-IDF) | Auto-discovers topics, handles noise |
| **Topic Labels** | TF-IDF keywords only | LLM-generated human-readable labels (Gemini) | From "credit card payment late" → "Credit Card Billing Disputes" |
| **Retrieval** | FAISS flat + raw query | + Cross-Encoder Reranking (two-stage funnel) | 18-42% quality gain in retrieval precision |
| **Evaluation** | None | Latency benchmarks (p50/p95/p99), cluster quality metrics | Quantified, not guessed |
| **Visualization** | Basic bar/pie charts | UMAP embedding space, radar charts, comparison dashboards | Interactive, explorable |

---

## Architecture

```mermaid
graph LR
    A[Raw CFPB Data<br/>200K complaints] --> B[Preprocessing<br/>Clean, normalize]
    B --> C{Multi-Model<br/>Embedding}
    C --> D1[MiniLM<br/>384d baseline]
    C --> D2[BGE<br/>768d SOTA]
    D1 & D2 --> E{Clustering}
    E --> F1[KMeans<br/>Fixed k=6]
    E --> F2[BERTopic<br/>Auto-k, HDBSCAN]
    D1 & D2 --> G{Retrieval}
    G --> H1[Vector FAISS]
    G --> H2[BM25 Sparse]
    G --> H3[Hybrid RRF]
    G --> H4[HyDE]
    G --> H5[Tree/RAPTOR]
    H3 --> I[Cross-Encoder<br/>Reranker]
    F1 & F2 --> J[Quality Metrics<br/>Silhouette, CH, DB]
    H1 & H2 & H3 & I --> K[Latency Benchmark<br/>p50, p95, p99]
    D1 & D2 --> L[UMAP<br/>2D Projection]
    J & K & L --> M[Streamlit<br/>Dashboard]
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Embeddings** | `sentence-transformers` | MiniLM (baseline) + BGE (SOTA) embedding generation |
| **Clustering** | `bertopic`, `hdbscan`, `umap-learn` | Modern topic modeling with density-based clustering |
| **Vector Search** | `faiss-cpu` | Fast approximate nearest neighbor search |
| **Sparse Retrieval** | `rank-bm25` | BM25 keyword-based retrieval |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder re-scoring for precision |
| **LLM** | Google Gemini | Topic labeling + complaint summarization |
| **Visualization** | `plotly`, `umap-learn` | Interactive UMAP plots + comparison charts |
| **Dashboard** | `streamlit` | 7-page interactive web application |
| **Data** | CFPB Consumer Complaints | 200K real-world financial complaints |

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/complaint-intelligence-system.git
cd complaint-intelligence-system

# Create virtual environment (Python 3.12 recommended)
uv venv .venv --python 3.12
# OR: python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
# OR: pip install -r requirements.txt
```

### 2. Get the Data

Download the CFPB Consumer Complaint Database:
- Source: [Consumer Financial Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- Place the CSV in `data/raw/complaints.csv`

### 3. Run the Pipeline

```bash
# Basic pipeline (MiniLM embeddings + KMeans clustering)
python run_pipeline.py

# Full showcase (both models + both clusterers + benchmarks)
python run_pipeline.py --model both --clustering both --benchmark

# With LLM summarization (requires GEMINI_API_KEY in .env)
python run_pipeline.py --model both --clustering both --benchmark --with-llm

# Custom sample size
python run_pipeline.py --sample-size 200000 --model both --clustering both --benchmark
```

### 4. Launch the Dashboard

```bash
streamlit run app/app.py
```

---

## Pipeline Flags

| Flag | Description | Default |
|---|---|---|
| `--sample-size N` | Number of complaints to process | 15000 |
| `--model {minilm,bge,both}` | Embedding model(s) to use | `minilm` |
| `--clustering {kmeans,bertopic,both}` | Clustering method(s) | `kmeans` |
| `--benchmark` | Run retrieval latency benchmarks | Off |
| `--with-llm` | Enable LLM summarization | Off |
| `--provider {gemini,groq,together}` | LLM provider | `gemini` |
| `--n-clusters N` | KMeans cluster count | 6 |

---

## Dashboard Pages

| Page | What It Shows |
|---|---|
| **Overview** | Key metrics, product/issue distributions, time trends |
| **Clusters** | Drill into clusters, see top products/issues per cluster |
| **Complaint Viewer** | Browse/filter individual complaints with AI summaries |
| **Ask AI** | Natural language Q&A over the complaint corpus (RAG) |
| **Embedding Comparison** | MiniLM vs BGE: throughput, similarity distributions, UMAP plots |
| **Clustering Comparison** | KMeans vs BERTopic: Silhouette, Calinski-Harabasz, Davies-Bouldin |
| **Retrieval Benchmark** | Latency comparison across Vector, BM25, Hybrid, Reranked |

---

## GPU Workflow (Colab Integration)

For large-scale runs (200K+ complaints), use the **official Google Colab VS Code extension**:

1. Install "Google Colab" extension in VS Code (by Google)
2. Open any `.ipynb` → Select Kernel → Colab → Sign in
3. Execute on cloud GPU (T4/A100), results stream to local filesystem

Your local RTX 3050 (4GB) is sufficient for runs under 50K texts.

---

## Project Structure

```
complaint-intelligence-system/
├── app/
│   └── app.py                          # Streamlit dashboard (7 pages)
├── src/
│   ├── preprocess.py                   # Text cleaning pipeline
│   ├── embeddings.py                   # Multi-model embedding engine
│   ├── embedding_benchmark.py          # Model comparison framework
│   ├── clustering.py                   # KMeans + BERTopic + quality metrics
│   ├── topic_labeler.py                # LLM topic label generation
│   ├── rag.py                          # RAG system with FAISS
│   ├── llm_utils.py                    # Multi-provider LLM utilities
│   ├── visualizer.py                   # UMAP + Plotly comparison plots
│   ├── retrievers/
│   │   ├── base.py                     # BaseRetriever interface
│   │   ├── vector_retriever.py         # FAISS dense retrieval
│   │   ├── bm25_retriever.py           # BM25 sparse retrieval
│   │   ├── hybrid_retriever.py         # RRF ensemble (Vector + BM25)
│   │   ├── hyde_retriever.py           # Hypothetical Document Embeddings
│   │   ├── tree_retriever.py           # RAPTOR-inspired hierarchical
│   │   ├── reranker.py                 # Cross-encoder reranking
│   │   └── reranked_retriever.py       # Two-stage retrieval wrapper
│   └── evaluation/
│       └── retrieval_benchmark.py      # Latency benchmarking framework
├── data/
│   ├── raw/                            # Raw CFPB data
│   ├── processed/                      # Processed data + embeddings
│   └── results/                        # Benchmark results (JSON)
├── notebooks/                          # Colab GPU notebooks
├── run_pipeline.py                     # End-to-end pipeline script
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Key Concepts Demonstrated

### 1. Multi-Model Embedding Comparison
The `EmbeddingRegistry` supports swapping models without changing downstream code. Compare encoding speed, cosine similarity distributions, and cluster separation across models.

### 2. BERTopic vs KMeans
BERTopic discovers the natural number of topics using HDBSCAN density-based clustering, while KMeans forces a fixed k. BERTopic also flags noisy/outlier documents rather than forcing them into clusters.

### 3. Two-Stage Retrieval Funnel
The modern production RAG pattern:
1. **Stage 1**: Hybrid retriever (Vector + BM25 + RRF) fetches top-50 broad candidates
2. **Stage 2**: Cross-encoder reranker re-scores with full cross-attention → returns top-5

This gives you the speed of bi-encoder retrieval with the precision of cross-encoder scoring.

### 4. Retrieval Benchmarking
Every retriever is measured on the same 20-query test set with p50/p95/p99 latency percentiles — the same metrics used in production systems.

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key_here
# Optional:
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

---

## License

MIT

---

## Author

Built by [Aswani Sahoo](https://github.com/AswaniSahoo) as a portfolio showcase for modern NLP techniques in complaint intelligence.
