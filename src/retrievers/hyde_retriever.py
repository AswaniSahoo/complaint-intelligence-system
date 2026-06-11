"""
HyDE retriever: generates a hypothetical complaint via LLM, embeds it,
then searches. Bridges the query-document vocabulary gap.
Ref: Gao et al., "Precise Zero-Shot Dense Retrieval" (2022).
"""

import logging
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd

from src.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

HYDE_PROMPT = """You are a helpful assistant. Given the following search query about customer complaints, 
write a realistic customer complaint (2-3 sentences) that would be relevant to this query.
Write ONLY the complaint text, nothing else.

Query: {query}

Hypothetical complaint:"""


class HyDERetriever(BaseRetriever):
    """HyDE: generate hypothetical document, embed it, then search."""

    name = "hyde"

    def __init__(self, embedder=None, llm_summarizer=None, embedding_dim=384):
        self.embedder = embedder
        self.llm = llm_summarizer
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.df: Optional[pd.DataFrame] = None

    def build_index(self, df, embeddings=None, **kwargs):
        """Build FAISS index from precomputed embeddings."""
        if embeddings is None:
            raise ValueError("HyDERetriever requires precomputed embeddings.")

        self.df = df.reset_index(drop=True)
        embeddings = embeddings.astype("float32").copy()
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        logger.info("HyDERetriever: built FAISS index with %d vectors", self.index.ntotal)

    def _generate_hypothesis(self, query: str) -> str:
        """Use LLM to generate a hypothetical complaint for the query."""
        if self.llm is None:
            # No LLM configured, degrade to raw query search
            logger.warning("HyDERetriever: no LLM configured, falling back to raw query")
            return query

        prompt = HYDE_PROMPT.format(query=query)

        try:
            if self.llm.provider == "gemini":
                response = self.llm.model.generate_content(prompt)
                return response.text.strip()
            else:
                # Groq / Together (OpenAI-compatible)
                response = self.llm.client.chat.completions.create(
                    model=self.llm.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("HyDERetriever: LLM hypothesis generation failed: %s", e)
            return query  # Fallback to raw query

    def retrieve(self, query, k=5):
        """Generate hypothesis, embed it, search FAISS."""
        if self.index is None or self.df is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not query or not query.strip():
            logger.warning("HyDERetriever: empty query received")
            return []
        if self.embedder is None:
            raise RuntimeError("HyDERetriever needs an embedder for encoding.")

        hypothesis = self._generate_hypothesis(query)
        logger.info("HyDERetriever: hypothesis = '%s'", hypothesis[:100])

        hyp_vec = self.embedder.encode([hypothesis], show_progress=False)[0]
        hyp_vec = hyp_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(hyp_vec)

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(hyp_vec, k)

        results: List[RetrievalResult] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0:
                continue
            row = self.df.iloc[idx]
            result = self._build_result(
                row, score=float(dist), rank=rank,
                extra_metadata={"hypothesis": hypothesis[:200]},
            )
            results.append(result)

        return results
