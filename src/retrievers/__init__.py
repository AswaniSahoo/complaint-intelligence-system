"""
Retrieval strategies for complaint search.

Provides multiple retrieval approaches:
    - VectorRetriever: FAISS dense semantic search
    - BM25Retriever: Sparse keyword-based retrieval
    - HybridRetriever: Ensemble of Vector + BM25 with Reciprocal Rank Fusion
    - HyDERetriever: Hypothetical Document Embeddings
    - TreeRetriever: PageIndex-inspired hierarchical reasoning-based retrieval
"""
