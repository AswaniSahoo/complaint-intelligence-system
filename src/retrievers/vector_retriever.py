"""Dense semantic search using FAISS (cosine similarity via IndexFlatIP)."""

import logging
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd

from src.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """FAISS-based dense vector retrieval."""

    name = "vector"

    def __init__(self, embedder=None, embedding_dim=384):
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.df: Optional[pd.DataFrame] = None

    def build_index(self, df, embeddings=None, **kwargs):
        """Build FAISS index from precomputed embeddings."""
        if embeddings is None:
            raise ValueError("VectorRetriever requires precomputed embeddings.")

        self.df = df.reset_index(drop=True)
        embeddings = embeddings.astype("float32").copy()
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        logger.info("VectorRetriever: built FAISS index with %d vectors", self.index.ntotal)

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k complaints by cosine similarity."""
        if self.index is None or self.df is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not query or not query.strip():
            logger.warning("VectorRetriever: empty query received")
            return []
        if self.embedder is None:
            raise RuntimeError("VectorRetriever needs an embedder for query encoding.")

        query_vec = self.embedder.encode([query], show_progress=False)[0]
        query_vec = query_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results: List[RetrievalResult] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0:
                continue
            row = self.df.iloc[idx]
            results.append(self._build_result(row, score=float(dist), rank=rank))

        return results
