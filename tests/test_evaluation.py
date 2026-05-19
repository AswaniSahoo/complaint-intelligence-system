"""Tests for the evaluation framework (Phase 5 placeholder + basic checks)."""

import pytest
from src.retrievers.base import RetrievalResult


class TestRetrievalResultSerialization:
    """Verify RetrievalResult round-trips through to_dict()."""

    def test_round_trip(self):
        original = RetrievalResult(
            text="complaint text",
            product="Credit card",
            issue="Billing",
            date="2024-01-01",
            score=0.95,
            rank=1,
            metadata={"cluster": 3},
        )
        d = original.to_dict()
        assert d["text"] == "complaint text"
        assert d["product"] == "Credit card"
        assert d["score"] == 0.95
        assert d["metadata"]["cluster"] == 3

    def test_empty_metadata(self):
        r = RetrievalResult(text="t")
        assert r.to_dict()["metadata"] == {}

    def test_multiple_results_unique_ranks(self):
        results = [
            RetrievalResult(text=f"r{i}", rank=i, score=1.0 / i)
            for i in range(1, 6)
        ]
        ranks = [r.rank for r in results]
        assert len(set(ranks)) == 5  # All unique


class TestRetrieverQAIntegration:
    """Test RetrieverQA wrapping a real retriever."""

    def test_answer_query_structure(self, tiny_df):
        from src.retrievers.bm25_retriever import BM25Retriever
        from src.rag import RetrieverQA

        bm25 = BM25Retriever()
        bm25.build_index(tiny_df)
        qa = RetrieverQA(bm25, tiny_df)

        resp = qa.answer_query("credit card", k=2)
        assert "query" in resp
        assert "retriever" in resp
        assert "results" in resp
        assert "count" in resp
        assert "latency_ms" in resp
        assert resp["retriever"] == "bm25"
        assert resp["count"] <= 2

    def test_get_insights_structure(self, tiny_df):
        from src.retrievers.bm25_retriever import BM25Retriever
        from src.rag import RetrieverQA

        bm25 = BM25Retriever()
        bm25.build_index(tiny_df)
        qa = RetrieverQA(bm25, tiny_df)

        insights = qa.get_insights("mortgage", k=3)
        assert "top_products" in insights
        assert "top_issues" in insights
        assert insights["total_found"] > 0
