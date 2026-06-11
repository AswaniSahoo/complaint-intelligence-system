"""Tests for src/visualizer.py | EmbeddingVisualizer."""

import pytest
import numpy as np
import plotly.graph_objects as go
from src.visualizer import EmbeddingVisualizer


class TestEmbeddingVisualizer:
    """Test visualization output structure (no UMAP computation needed)."""

    @pytest.fixture
    def viz(self, tmp_path):
        return EmbeddingVisualizer(cache_dir=str(tmp_path))

    @pytest.fixture
    def fake_projection(self):
        rng = np.random.RandomState(42)
        return rng.randn(50, 2).astype(np.float32)

    def test_plot_embedding_space_returns_figure(self, viz, fake_projection):
        labels = [i % 3 for i in range(50)]
        fig = viz.plot_embedding_space(fake_projection, labels)
        assert isinstance(fig, go.Figure)

    def test_plot_embedding_space_with_hover(self, viz, fake_projection):
        labels = [i % 3 for i in range(50)]
        hover_texts = [f"text_{i}" for i in range(50)]
        fig = viz.plot_embedding_space(
            fake_projection, labels, hover_texts=hover_texts
        )
        assert isinstance(fig, go.Figure)

    def test_plot_model_comparison(self, viz, fake_projection):
        projections = {
            "minilm": fake_projection,
            "bge": fake_projection + 1,
        }
        labels = [i % 3 for i in range(50)]
        model_names = {"minilm": "MiniLM-L6-v2", "bge": "BGE-base-en-v1.5"}
        fig = viz.plot_model_comparison(projections, labels, model_names)
        assert isinstance(fig, go.Figure)

    def test_plot_retrieval_comparison(self, viz):
        benchmark_results = {
            "vector": {
                "latency": {"p50_ms": 5.0, "p95_ms": 12.0, "p99_ms": 20.0, "mean_ms": 7.0},
                "avg_results_returned": 5,
            },
            "bm25": {
                "latency": {"p50_ms": 2.0, "p95_ms": 5.0, "p99_ms": 8.0, "mean_ms": 3.0},
                "avg_results_returned": 5,
            },
        }
        fig = viz.plot_retrieval_comparison(benchmark_results)
        assert isinstance(fig, go.Figure)

    def test_plot_retrieval_comparison_handles_errors(self, viz):
        """Entries with 'error' key should be skipped."""
        benchmark_results = {
            "vector": {
                "latency": {"p50_ms": 5.0, "p95_ms": 12.0, "p99_ms": 20.0, "mean_ms": 7.0},
                "avg_results_returned": 5,
            },
            "broken": {"error": "failed to build index"},
        }
        fig = viz.plot_retrieval_comparison(benchmark_results)
        assert isinstance(fig, go.Figure)

    def test_plot_cluster_comparison(self, viz):
        comparison_results = {
            "kmeans": {
                "method": "KMeans(k=6)",
                "n_clusters": 6,
                "silhouette": 0.15,
                "davies_bouldin": 2.3,
            },
            "bertopic": {
                "method": "BERTopic",
                "n_clusters": 12,
                "silhouette": 0.22,
                "davies_bouldin": 1.8,
            },
        }
        fig = viz.plot_cluster_comparison(comparison_results)
        assert isinstance(fig, go.Figure)

    def test_umap_cache_roundtrip(self, viz, tmp_path):
        """Test UMAP projection caching to disk."""
        rng = np.random.RandomState(0)
        fake_emb = rng.randn(20, 8).astype(np.float32)
        # First call: compute and cache
        proj = viz.generate_umap_projection(
            fake_emb, cache_key="test_cache"
        )
        assert proj.shape == (20, 2)

        # Second call: load from cache
        proj2 = viz.generate_umap_projection(
            fake_emb, cache_key="test_cache"
        )
        np.testing.assert_array_equal(proj, proj2)
