"""
BM25 Retriever — Sparse keyword-based retrieval.

Uses the Okapi BM25 algorithm for term-frequency-based ranking.
No vectors needed — purely lexical matching. Good baseline and
excels on keyword-heavy / exact-match queries.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing."""
    if not text or not isinstance(text, str):
        return []
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    """BM25-based sparse retrieval (vectorless)."""

    name = "bm25"

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.df: Optional[pd.DataFrame] = None
        self.corpus_tokens: Optional[List[List[str]]] = None

    def build_index(self, df: pd.DataFrame, embeddings=None, **kwargs):
        """
        Build BM25 index from the clean_text column.

        Args:
            df: Complaints DataFrame (must have 'clean_text' column).
            embeddings: Ignored — BM25 is vectorless.
        """
        text_column = kwargs.get("text_column", "clean_text")
        self.df = df.reset_index(drop=True)

        logger.info("BM25Retriever: tokenizing %d documents...", len(self.df))
        self.corpus_tokens = [tokenize(str(text)) for text in self.df[text_column]]

        # Filter out empty token lists to avoid BM25 issues
        valid_mask = [len(tokens) > 0 for tokens in self.corpus_tokens]
        if not all(valid_mask):
            n_empty = sum(1 for v in valid_mask if not v)
            logger.warning("BM25Retriever: %d documents had empty tokens (skipped in index)", n_empty)

        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info("BM25Retriever: index built with %d documents", len(self.corpus_tokens))

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k complaints by BM25 score."""
        if self.bm25 is None or self.df is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not query or not query.strip():
            logger.warning("BM25Retriever: empty query received")
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        k = min(k, len(self.df))
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0:
                break  # No point returning zero-score results
            row = self.df.iloc[idx]
            results.append(self._build_result(row, score=score, rank=rank))

        return results
