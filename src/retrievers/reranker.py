"""
Cross-encoder reranker for second-pass rescoring.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (runs locally, no API key).
"""

import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-score retrieval candidates using a cross-encoder."""

    def __init__(self, model_name=DEFAULT_RERANKER):
        print(f"Loading cross-encoder reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print("Cross-encoder loaded (runs locally, no API key needed)")

    def rerank(self, query, candidate_texts, top_k=None):
        """Re-score and sort candidates by relevance. Returns [(idx, score, text)]."""
        if not candidate_texts:
            return []

        pairs = [(query, text) for text in candidate_texts]

        scores = self.model.predict(pairs)

        indexed_results = [
            (i, float(score), text)
            for i, (score, text) in enumerate(zip(scores, candidate_texts))
        ]
        indexed_results.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_results = indexed_results[:top_k]

        return indexed_results
