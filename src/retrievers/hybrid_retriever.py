"""Hybrid retriever: Vector + BM25 merged via Reciprocal Rank Fusion (RRF)."""

import logging
from typing import List, Dict

import pandas as pd
import numpy as np

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult]:
    """Merge ranked lists using RRF. Score = sum(1 / (k + rank))."""
    scores = {}
    best_result: Dict[str, RetrievalResult] = {}

    for result_list in ranked_lists:
        for result in result_list:
            doc_key = result.text[:200]  # Use first 200 chars as dedup key
            rrf_score = 1.0 / (k + result.rank)
            scores[doc_key] = scores.get(doc_key, 0.0) + rrf_score

            # Keep the result object with the higher individual score
            if doc_key not in best_result or result.score > best_result[doc_key].score:
                best_result[doc_key] = result

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results: List[RetrievalResult] = []
    for rank, key in enumerate(sorted_keys, start=1):
        result = best_result[key]
        result.score = scores[key]
        result.rank = rank
        result.metadata["rrf_score"] = scores[key]
        fused_results.append(result)

    return fused_results


class HybridRetriever(BaseRetriever):
    """Ensemble retriever combining Vector + BM25 via Reciprocal Rank Fusion."""

    name = "hybrid"

    def __init__(self, vector_retriever, bm25_retriever, rrf_k=60):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def build_index(self, df, embeddings=None, **kwargs):
        """Build indices for both sub-retrievers."""
        self.vector_retriever.build_index(df, embeddings, **kwargs)
        self.bm25_retriever.build_index(df, **kwargs)
        logger.info("HybridRetriever: both sub-indices built")

    def retrieve(self, query, k=5):
        """Retrieve top-k by fusing Vector + BM25 results with RRF."""
        if not query or not query.strip():
            logger.warning("HybridRetriever: empty query received")
            return []

        candidate_k = min(k * 2, 50)  # Fetch more candidates for fusion

        vector_results = self.vector_retriever.retrieve(query, k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query, k=candidate_k)

        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.rrf_k,
        )

        return fused[:k]
