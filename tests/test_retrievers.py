"""Tests for all retriever implementations."""

import pytest
import numpy as np
from src.retrievers.base import RetrievalResult
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.tree_retriever import TreeRetriever


# ---------- RetrievalResult dataclass -----------------------------------------

class TestRetrievalResult:

    def test_to_dict(self):
        r = RetrievalResult(text="test", product="P", issue="I", score=0.9, rank=1)
        d = r.to_dict()
        assert d["text"] == "test"
        assert d["score"] == 0.9
        assert isinstance(d["metadata"], dict)

    def test_defaults(self):
        r = RetrievalResult(text="x")
        assert r.product == "Unknown"
        assert r.score == 0.0
        assert r.metadata == {}


# ---------- BM25 Retriever (no GPU / no embedder needed) ----------------------

class TestBM25Retriever:

    @pytest.fixture(autouse=True)
    def setup(self, tiny_df):
        self.retriever = BM25Retriever()
        self.retriever.build_index(tiny_df)

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("credit card", k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_result_type(self):
        results = self.retriever.retrieve("mortgage", k=1)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_result_has_required_fields(self):
        results = self.retriever.retrieve("bank account", k=1)
        if results:
            r = results[0]
            assert r.text != ""
            assert r.product != ""
            assert r.rank >= 1
            assert r.score > 0

    def test_empty_query_returns_empty(self):
        assert self.retriever.retrieve("", k=3) == []

    def test_k_zero_returns_empty(self):
        assert self.retriever.retrieve("test", k=0) == []

    def test_k_larger_than_dataset(self):
        results = self.retriever.retrieve("card", k=100)
        assert len(results) <= 5  # tiny_df has 5 rows

    def test_scores_descending(self):
        results = self.retriever.retrieve("credit card fees", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_with_timing(self):
        results, latency = self.retriever.retrieve_with_timing("test", k=2)
        assert isinstance(latency, float)
        assert latency >= 0


# ---------- Tree Retriever (no LLM — fallback mode) --------------------------

class TestTreeRetriever:

    @pytest.fixture(autouse=True)
    def setup(self, tiny_df):
        self.retriever = TreeRetriever(llm_summarizer=None)
        self.retriever.build_index(tiny_df)

    def test_tree_built(self):
        assert self.retriever.root is not None
        assert len(self.retriever.root.children) > 0

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("credit card billing", k=2)
        assert len(results) > 0

    def test_empty_query(self):
        assert self.retriever.retrieve("", k=3) == []

    def test_result_has_metadata(self):
        results = self.retriever.retrieve("mortgage", k=1)
        if results:
            assert "retrieval_path" in results[0].metadata

    def test_not_built_raises(self):
        fresh = TreeRetriever()
        with pytest.raises(RuntimeError):
            fresh.retrieve("test", k=1)


# ---------- Retriever not-built guard -----------------------------------------

class TestRetrieverNotBuilt:

    def test_bm25_not_built(self):
        with pytest.raises(RuntimeError):
            BM25Retriever().retrieve("test", k=1)

    def test_tree_not_built(self):
        with pytest.raises(RuntimeError):
            TreeRetriever().retrieve("test", k=1)
