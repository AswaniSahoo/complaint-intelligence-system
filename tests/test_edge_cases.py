"""Edge case tests | null inputs, malformed data, boundary conditions."""

import pytest
import pandas as pd
import numpy as np
from src.preprocess import clean_text
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.tree_retriever import TreeRetriever


class TestCleanTextEdgeCases:

    def test_only_whitespace(self):
        assert clean_text("   \t\n  ") == ""

    def test_only_url(self):
        result = clean_text("http://example.com")
        assert result == "" or "http" not in result

    def test_only_email(self):
        result = clean_text("user@domain.com")
        assert result == "" or "@" not in result

    def test_only_special_chars(self):
        result = clean_text("@#$%^&*()")
        assert result == "" or all(c.isalnum() or c in " .,!?-" for c in result)

    def test_extremely_long_single_word(self):
        word = "a" * 100000
        result = clean_text(word)
        assert len(result) == 100000

    def test_boolean_input(self):
        result = clean_text(True)
        assert isinstance(result, str)

    def test_list_input(self):
        result = clean_text([1, 2, 3])
        assert isinstance(result, str)


class TestBM25EdgeCases:

    @pytest.fixture(autouse=True)
    def setup(self, tiny_df):
        self.retriever = BM25Retriever()
        self.retriever.build_index(tiny_df)

    def test_whitespace_only_query(self):
        assert self.retriever.retrieve("   ", k=3) == []

    def test_special_chars_query(self):
        # Should not crash
        results = self.retriever.retrieve("@#$%^&", k=3)
        assert isinstance(results, list)

    def test_very_long_query(self):
        long_query = "credit card " * 500
        results = self.retriever.retrieve(long_query, k=2)
        assert isinstance(results, list)

    def test_single_char_query(self):
        results = self.retriever.retrieve("a", k=2)
        assert isinstance(results, list)

    def test_numeric_query(self):
        results = self.retriever.retrieve("12345", k=2)
        assert isinstance(results, list)


class TestTreeEdgeCases:

    def test_build_with_missing_cluster_column(self):
        df_no_cluster = pd.DataFrame({
            "complaint_text": ["test complaint"],
            "clean_text": ["test complaint"],
            "product": ["Test Product"],
            "issue": ["Test Issue"],
            "date": [pd.Timestamp("2024-01-01")],
        })
        tree = TreeRetriever(llm_summarizer=None)
        tree.build_index(df_no_cluster)
        assert tree.root is not None

    def test_build_with_nan_values(self):
        df_nan = pd.DataFrame({
            "complaint_text": ["test", None],
            "clean_text": ["test", ""],
            "product": ["P1", np.nan],
            "issue": [np.nan, "I2"],
            "date": [pd.NaT, pd.Timestamp("2024-01-01")],
            "cluster": [0, np.nan],
        })
        tree = TreeRetriever(llm_summarizer=None)
        tree.build_index(df_nan)
        assert tree.root is not None

    def test_single_row_dataset(self):
        df_single = pd.DataFrame({
            "complaint_text": ["one complaint"],
            "clean_text": ["one complaint"],
            "product": ["Product"],
            "issue": ["Issue"],
            "date": [pd.Timestamp("2024-01-01")],
            "cluster": [0],
        })
        tree = TreeRetriever(llm_summarizer=None)
        tree.build_index(df_single)
        results = tree.retrieve("complaint", k=1)
        assert isinstance(results, list)


class TestBM25WithMalformedData:

    def test_empty_dataframe(self):
        df_empty = pd.DataFrame({
            "clean_text": [],
            "complaint_text": [],
            "product": [],
            "issue": [],
            "date": [],
        })
        retriever = BM25Retriever()
        retriever.build_index(df_empty)
        results = retriever.retrieve("test", k=3)
        assert results == []

    def test_all_empty_texts(self):
        df_empty_texts = pd.DataFrame({
            "clean_text": ["", "", ""],
            "complaint_text": ["", "", ""],
            "product": ["P", "P", "P"],
            "issue": ["I", "I", "I"],
            "date": ["2024-01-01"] * 3,
        })
        retriever = BM25Retriever()
        retriever.build_index(df_empty_texts)
        results = retriever.retrieve("test", k=2)
        assert isinstance(results, list)
