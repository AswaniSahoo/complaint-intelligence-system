"""
Retrieval strategies for complaint search.

Provides multiple retrieval approaches:
    - VectorRetriever: FAISS dense semantic search
    - BM25Retriever: Sparse keyword-based retrieval
    - HybridRetriever: Ensemble of Vector + BM25 with Reciprocal Rank Fusion
    - HyDERetriever: Hypothetical Document Embeddings
    - TreeRetriever: PageIndex-inspired hierarchical reasoning-based retrieval
    - RerankedRetriever: Wraps any retriever with cross-encoder reranking
"""

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.hyde_retriever import HyDERetriever
from src.retrievers.tree_retriever import TreeRetriever
from src.retrievers.reranked_retriever import RerankedRetriever

ALL_RETRIEVERS = {
    "vector": VectorRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
    "hyde": HyDERetriever,
    "tree": TreeRetriever,
    "reranked": RerankedRetriever,
}

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "VectorRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "HyDERetriever",
    "TreeRetriever",
    "RerankedRetriever",
    "ALL_RETRIEVERS",
]
