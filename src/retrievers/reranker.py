"""
Cross-Encoder Reranker — The single biggest retrieval quality improvement.

Uses a cross-encoder model to re-score query-document pairs by processing
them jointly (full cross-attention), rather than independently as bi-encoders
do. This provides much higher precision at the cost of not being able to
search the full corpus (hence used as a second-pass reranker).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Lightweight (22M params), fast inference
    - No API key required — runs entirely locally
    - Trained on MS MARCO passage ranking

Reference:
    - Industry benchmarks show 18-42% retrieval quality improvement
      from adding a cross-encoder reranking stage.
"""

import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-score and re-sort retrieval candidates using a cross-encoder.

    Usage::

        reranker = CrossEncoderReranker()
        scored = reranker.rerank("billing issue", candidate_texts, top_k=5)
        # scored = [(index, score, text), ...]
    """

    def __init__(self, model_name=DEFAULT_RERANKER):
        """
        Args:
            model_name: HuggingFace cross-encoder model name.
        """
        print(f"Loading cross-encoder reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print("Cross-encoder loaded (runs locally, no API key needed)")

    def rerank(self, query, candidate_texts, top_k=None):
        """
        Re-score candidates and return them sorted by relevance.

        Args:
            query: The search query string.
            candidate_texts: List of candidate document texts.
            top_k: Return only top-k results. None = return all, re-sorted.

        Returns:
            List of (original_index, score, text) tuples, sorted by
            descending relevance score.
        """
        if not candidate_texts:
            return []

        # Build query-document pairs for cross-encoder
        pairs = [(query, text) for text in candidate_texts]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Build indexed results and sort by score descending
        indexed_results = [
            (i, float(score), text)
            for i, (score, text) in enumerate(zip(scores, candidate_texts))
        ]
        indexed_results.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_results = indexed_results[:top_k]

        return indexed_results
