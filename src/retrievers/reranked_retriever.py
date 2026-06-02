"""
Reranked Retriever — Wraps any BaseRetriever with cross-encoder reranking.

Two-stage retrieval funnel:
    1. Base retriever fetches a wide set of candidates (top-50)
    2. Cross-encoder reranks them for precision (returns top-k)

This is the modern production RAG pattern that consistently delivers
the largest single improvement in retrieval quality.
"""

import logging
from typing import List

import pandas as pd
import numpy as np

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.retrievers.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RerankedRetriever(BaseRetriever):
    """Wraps any retriever and adds cross-encoder reranking as post-processing."""

    name = "reranked"

    def __init__(self, base_retriever: BaseRetriever,
                 reranker: CrossEncoderReranker = None,
                 candidate_k: int = 50):
        """
        Args:
            base_retriever: Any BaseRetriever implementation to use for
                            initial candidate retrieval.
            reranker: CrossEncoderReranker instance. If None, one is
                      created with the default model.
            candidate_k: Number of candidates to retrieve before reranking.
        """
        self.base_retriever = base_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_k = candidate_k
        self.name = f"reranked_{base_retriever.name}"

    def build_index(self, df: pd.DataFrame, embeddings: np.ndarray = None, **kwargs):
        """Delegate index building to the base retriever."""
        self.base_retriever.build_index(df, embeddings, **kwargs)
        logger.info(
            "RerankedRetriever: base index built (%s), reranker ready",
            self.base_retriever.name
        )

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Two-stage retrieval: base retriever → cross-encoder reranking.

        1. Fetch candidate_k results from the base retriever
        2. Re-score with cross-encoder
        3. Return top-k by reranked score
        """
        if not query or not query.strip():
            logger.warning("RerankedRetriever: empty query received")
            return []

        # Stage 1: Get candidates from base retriever
        candidates = self.base_retriever.retrieve(query, k=self.candidate_k)

        if not candidates:
            return []

        # Stage 2: Rerank with cross-encoder
        candidate_texts = [r.text for r in candidates]
        reranked = self.reranker.rerank(query, candidate_texts, top_k=k)

        # Build results with updated scores and ranks
        results: List[RetrievalResult] = []
        for new_rank, (orig_idx, ce_score, _text) in enumerate(reranked, start=1):
            original = candidates[orig_idx]
            # Preserve original metadata, update score and rank
            result = RetrievalResult(
                text=original.text,
                product=original.product,
                issue=original.issue,
                date=original.date,
                score=ce_score,
                rank=new_rank,
                metadata={
                    **original.metadata,
                    "pre_rerank_score": original.score,
                    "pre_rerank_rank": original.rank,
                    "reranker_model": self.reranker.model_name,
                },
            )
            results.append(result)

        return results
