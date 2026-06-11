"""Two-stage retriever: base retriever for candidates, cross-encoder for precision."""

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

    def __init__(self, base_retriever, reranker=None, candidate_k=50):
        self.base_retriever = base_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_k = candidate_k
        self.name = f"reranked_{base_retriever.name}"

    def build_index(self, df, embeddings=None, **kwargs):
        """Delegate index building to the base retriever."""
        self.base_retriever.build_index(df, embeddings, **kwargs)
        logger.info(
            "RerankedRetriever: base index built (%s), reranker ready",
            self.base_retriever.name
        )

    def retrieve(self, query, k=5):
        """Fetch candidates from base retriever, then rerank with cross-encoder."""
        if not query or not query.strip():
            logger.warning("RerankedRetriever: empty query received")
            return []

        candidates = self.base_retriever.retrieve(query, k=self.candidate_k)

        if not candidates:
            return []

        candidate_texts = [r.text for r in candidates]
        reranked = self.reranker.rerank(query, candidate_texts, top_k=k)

        results = []
        for new_rank, (orig_idx, ce_score, _text) in enumerate(reranked, start=1):
            original = candidates[orig_idx]
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
