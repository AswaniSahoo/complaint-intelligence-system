"""Tests for src/retrievers/reranker.py and reranked_retriever.py."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.retrievers.base import RetrievalResult
from src.retrievers.reranked_retriever import RerankedRetriever


class TestRerankedRetrieverLogic:
    """Test RerankedRetriever without loading cross-encoder model."""

    @pytest.fixture
    def mock_base_retriever(self, tiny_df):
        """Create a mock base retriever."""
        from src.retrievers.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()
        retriever.build_index(tiny_df)
        return retriever

    @pytest.fixture
    def mock_reranker(self):
        """Create a mock reranker that returns deterministic scores."""
        reranker = MagicMock()
        reranker.model_name = "mock-reranker"

        def fake_rerank(query, candidate_texts, top_k=None):
            # Return candidates in reverse order with fake scores
            results = [
                (i, float(len(candidate_texts) - i), text)
                for i, text in enumerate(candidate_texts)
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            if top_k:
                results = results[:top_k]
            return results

        reranker.rerank = fake_rerank
        return reranker

    def test_empty_query_returns_empty(self, mock_base_retriever, mock_reranker):
        rr = RerankedRetriever(mock_base_retriever, mock_reranker, candidate_k=10)
        assert rr.retrieve("", k=3) == []

    def test_whitespace_query_returns_empty(self, mock_base_retriever, mock_reranker):
        rr = RerankedRetriever(mock_base_retriever, mock_reranker, candidate_k=10)
        assert rr.retrieve("   ", k=3) == []

    def test_retrieve_returns_results(self, mock_base_retriever, mock_reranker):
        rr = RerankedRetriever(mock_base_retriever, mock_reranker, candidate_k=10)
        results = rr.retrieve("credit card", k=2)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_have_reranker_metadata(self, mock_base_retriever, mock_reranker):
        rr = RerankedRetriever(mock_base_retriever, mock_reranker, candidate_k=10)
        results = rr.retrieve("mortgage", k=2)
        if results:
            r = results[0]
            assert "reranker_model" in r.metadata
            assert "pre_rerank_score" in r.metadata
            assert "pre_rerank_rank" in r.metadata
            assert r.metadata["reranker_model"] == "mock-reranker"

    def test_ranks_are_sequential(self, mock_base_retriever, mock_reranker):
        rr = RerankedRetriever(mock_base_retriever, mock_reranker, candidate_k=10)
        results = rr.retrieve("credit card fees", k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_name_includes_base(self, mock_base_retriever, mock_reranker):
        rr = RerankedRetriever(mock_base_retriever, mock_reranker)
        assert "bm25" in rr.name
        assert "reranked" in rr.name
