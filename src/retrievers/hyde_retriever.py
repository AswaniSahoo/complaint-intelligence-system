"""
HyDE Retriever — Hypothetical Document Embeddings.

Instead of embedding the raw query, HyDE asks an LLM to generate a
*hypothetical* complaint that would answer the query, then embeds THAT
hypothetical document and uses it for vector search.

This bridges the query-document vocabulary gap and often outperforms
naive vector search for abstract or short queries.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (2022).
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

    def __init__(self, embedder=None, llm_summarizer=None, embedding_dim: int = 384):
        """
        Args:
            embedder: ComplaintEmbedder instance for encoding.
            llm_summarizer: LLMSummarizer instance for hypothesis generation.
            embedding_dim: Embedding vector dimension.
        """
        self.embedder = embedder
        self.llm = llm_summarizer
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.df: Optional[pd.DataFrame] = None

    def build_index(self, df: pd.DataFrame, embeddings: np.ndarray = None, **kwargs):
        """
        Build FAISS index from precomputed embeddings.

        Args:
            df: Complaints DataFrame.
            embeddings: Precomputed embeddings array.
        """
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
            # Fallback: just return the query itself (degrades to regular vector search)
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

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Generate a hypothetical complaint, embed it, and search.

        Steps:
            1. LLM generates a hypothetical complaint from the query.
            2. Embed the hypothetical complaint.
            3. Search FAISS index with that embedding.
        """
        if self.index is None or self.df is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not query or not query.strip():
            logger.warning("HyDERetriever: empty query received")
            return []
        if self.embedder is None:
            raise RuntimeError("HyDERetriever needs an embedder for encoding.")

        # Step 1: Generate hypothesis
        hypothesis = self._generate_hypothesis(query)
        logger.info("HyDERetriever: hypothesis = '%s'", hypothesis[:100])

        # Step 2: Embed the hypothesis
        hyp_vec = self.embedder.encode([hypothesis], show_progress=False)[0]
        hyp_vec = hyp_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(hyp_vec)

        # Step 3: Search
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
