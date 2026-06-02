"""
Complaint Intelligence System - Core Package.

Modules:
    preprocess: Text cleaning and data preparation
    embeddings: Multi-model embedding generation (MiniLM, BGE)
    embedding_benchmark: Head-to-head embedding model comparison
    clustering: KMeans (baseline) and BERTopic (SOTA) clustering
    topic_labeler: LLM-generated topic labels for clusters
    llm_utils: Multi-provider LLM summarization (Gemini/Groq/Together)
    rag: Retrieval-Augmented Generation with FAISS
    visualizer: UMAP projections and comparison plots
    retrievers: Vector, BM25, Hybrid, HyDE, Tree, Reranked retrieval
    evaluation: Retrieval benchmarking framework
"""
