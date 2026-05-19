"""
Hybrid Retriever — Ensemble of Vector + BM25 with Reciprocal Rank Fusion.

Combines the strengths of dense semantic search (captures meaning) and
sparse keyword search (captures exact terms) by merging their ranked
result lists using RRF.  This is one of the most effective retrieval
strategies in practice.
"""

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
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over lists L: 1 / (k + rank_L(d))

    Args:
        ranked_lists: List of result lists from different retrievers.
        k: RRF constant (default 60, as in the original paper).

    Returns:
        Merged and re-ranked list of RetrievalResult.
    """
    # Map document text → aggregated RRF score and best result object
    scores: Dict[str, float] = {}
    best_result: Dict[str, RetrievalResult] = {}

    for result_list in ranked_lists:
        for result in result_list:
            doc_key = result.text[:200]  # Use first 200 chars as dedup key
            rrf_score = 1.0 / (k + result.rank)
            scores[doc_key] = scores.get(doc_key, 0.0) + rrf_score

            # Keep the result object with the higher individual score
            if doc_key not in best_result or result.score > best_result[doc_key].score:
                best_result[doc_key] = result

    # Sort by fused score descending
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

    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever,
                 rrf_k: int = 60):
        """
        Args:
            vector_retriever: Pre-configured VectorRetriever instance.
            bm25_retriever: Pre-configured BM25Retriever instance.
            rrf_k: RRF constant (default 60).
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def build_index(self, df: pd.DataFrame, embeddings: np.ndarray = None, **kwargs):
        """
        Build indices for both sub-retrievers.

        Args:
            df: Complaints DataFrame.
            embeddings: Precomputed embeddings (passed to VectorRetriever).
        """
        self.vector_retriever.build_index(df, embeddings, **kwargs)
        self.bm25_retriever.build_index(df, **kwargs)
        logger.info("HybridRetriever: both sub-indices built")

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve top-k by running both retrievers and fusing with RRF.

        Each sub-retriever fetches 2*k candidates to ensure enough
        diversity after fusion.
        """
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
