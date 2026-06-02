"""Tests for src/clustering.py — KMeansClusterer, ClusterComparison."""

import pytest
import numpy as np
import pandas as pd
from src.clustering import KMeansClusterer, ClusterComparison


class TestKMeansClusterer:
    """Test KMeans clustering logic."""

    @pytest.fixture
    def dummy_embeddings(self):
        """Small random embeddings for testing."""
        rng = np.random.RandomState(42)
        return rng.randn(50, 16).astype(np.float32)

    def test_fit_predict_shape(self, dummy_embeddings):
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        labels = clusterer.fit_predict(dummy_embeddings)
        assert len(labels) == 50

    def test_fit_predict_label_range(self, dummy_embeddings):
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        labels = clusterer.fit_predict(dummy_embeddings)
        assert set(labels).issubset({0, 1, 2})

    def test_labels_stored(self, dummy_embeddings):
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        labels = clusterer.fit_predict(dummy_embeddings)
        assert clusterer.cluster_labels is not None
        np.testing.assert_array_equal(labels, clusterer.cluster_labels)

    def test_extract_keywords_returns_dict(self, tiny_df):
        """Test extract_keywords with tiny data."""
        rng = np.random.RandomState(42)
        emb = rng.randn(len(tiny_df), 16).astype(np.float32)
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        labels = clusterer.fit_predict(emb)
        tiny_df = tiny_df.copy()
        tiny_df["cluster"] = labels
        keywords = clusterer.extract_keywords(tiny_df, text_column="clean_text", top_n=3)
        assert isinstance(keywords, dict)
        assert len(keywords) == 2

    def test_save_load_roundtrip(self, dummy_embeddings, tmp_path):
        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        clusterer.fit_predict(dummy_embeddings)
        model_path = str(tmp_path / "kmeans.pkl")
        clusterer.save_model(model_path)

        new_clusterer = KMeansClusterer(n_clusters=2)
        new_clusterer.load_model(model_path)
        assert new_clusterer.kmeans is not None


class TestClusterComparison:
    """Test cluster quality metric computation."""

    @pytest.fixture
    def embeddings_and_labels(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(100, 16).astype(np.float32)
        labels_a = np.array([i % 3 for i in range(100)])
        labels_b = np.array([i % 5 for i in range(100)])
        return emb, labels_a, labels_b

    def test_compare_returns_both_methods(self, embeddings_and_labels):
        emb, labels_a, labels_b = embeddings_and_labels
        result = ClusterComparison.compare(emb, labels_a, labels_b)
        assert "kmeans" in result
        assert "bertopic" in result

    def test_compare_has_required_metrics(self, embeddings_and_labels):
        emb, labels_a, labels_b = embeddings_and_labels
        result = ClusterComparison.compare(emb, labels_a, labels_b)
        for key in ["kmeans", "bertopic"]:
            m = result[key]
            assert "method" in m
            assert "n_clusters" in m
            assert "silhouette" in m
            assert "calinski_harabasz" in m
            assert "davies_bouldin" in m

    def test_silhouette_in_valid_range(self, embeddings_and_labels):
        emb, labels_a, labels_b = embeddings_and_labels
        result = ClusterComparison.compare(emb, labels_a, labels_b)
        for key in ["kmeans", "bertopic"]:
            sil = result[key]["silhouette"]
            if sil is not None:
                assert -1.0 <= sil <= 1.0

    def test_compute_metrics_with_outlier_labels(self):
        """BERTopic produces -1 labels for outliers."""
        rng = np.random.RandomState(1)
        emb = rng.randn(30, 8).astype(np.float32)
        labels = np.array([-1, -1, -1] + [i % 3 for i in range(27)])
        result = ClusterComparison.compute_metrics(emb, labels, "test_outliers")
        assert result["n_outliers"] == 3

    def test_compute_metrics_single_cluster_returns_none(self):
        """Single valid cluster cannot compute silhouette."""
        rng = np.random.RandomState(2)
        emb = rng.randn(10, 4).astype(np.float32)
        labels = np.zeros(10, dtype=int)
        result = ClusterComparison.compute_metrics(emb, labels, "single")
        assert result["silhouette"] is None
        assert result["n_clusters"] == 1
