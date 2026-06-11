"""
UMAP projections and comparison plots for embeddings and retrieval results.
All plots use Plotly for Streamlit integration. UMAP projections are cached.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """Generate UMAP projections and comparison plots for embeddings."""

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir

    def generate_umap_projection(self, embeddings, n_neighbors=15,
                                 min_dist=0.1, random_state=42,
                                 cache_key=None):
        """Reduce embeddings to 2D using UMAP. Caches results if cache_key given."""

        if cache_key and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"umap_{cache_key}.npy")
            if os.path.exists(cache_path):
                projection = np.load(cache_path)
                print(f"Loaded cached UMAP projection: {cache_key}")
                return projection

        from umap import UMAP

        print(f"Computing UMAP projection for {embeddings.shape}...")
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
        )
        projection = reducer.fit_transform(embeddings)
        print(f"UMAP complete: {projection.shape}")


        if cache_key and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_path, projection)
            print(f"Cached UMAP projection: {cache_path}")

        return projection

    def plot_embedding_space(self, projection, labels, label_name="Cluster",
                             title="Embedding Space (UMAP)", hover_texts=None):
        """2D scatter plot of the embedding space."""
        df_plot = pd.DataFrame({
            "UMAP-1": projection[:, 0],
            "UMAP-2": projection[:, 1],
            label_name: [str(l) for l in labels],
        })

        if hover_texts is not None:
            df_plot["Text"] = [t[:100] + "..." if len(t) > 100 else t
                               for t in hover_texts]
            hover_data = ["Text"]
        else:
            hover_data = None

        fig = px.scatter(
            df_plot, x="UMAP-1", y="UMAP-2",
            color=label_name,
            title=title,
            hover_data=hover_data,
            opacity=0.6,
            width=800, height=600,
        )
        fig.update_layout(
            template="plotly_dark",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        )
        fig.update_traces(marker=dict(size=3))

        return fig

    def plot_model_comparison(self, projections, labels, model_names,
                              label_name="Cluster"):
        """Side-by-side UMAP subplots comparing embedding models."""
        keys = list(projections.keys())
        fig = make_subplots(
            rows=1, cols=len(keys),
            subplot_titles=[model_names.get(k, k) for k in keys],
            horizontal_spacing=0.08,
        )

        unique_labels = sorted(set(str(l) for l in labels))
        colors = px.colors.qualitative.Set2
        color_map = {label: colors[i % len(colors)]
                     for i, label in enumerate(unique_labels)}

        for col_idx, key in enumerate(keys, start=1):
            proj = projections[key]
            for label in unique_labels:
                mask = np.array([str(l) == label for l in labels])
                fig.add_trace(
                    go.Scatter(
                        x=proj[mask, 0], y=proj[mask, 1],
                        mode="markers",
                        marker=dict(size=2, color=color_map[label], opacity=0.5),
                        name=f"{label_name} {label}",
                        showlegend=(col_idx == 1),
                        legendgroup=label,
                    ),
                    row=1, col=col_idx,
                )

        fig.update_layout(
            title="Embedding Model Comparison (UMAP 2D)",
            template="plotly_dark",
            height=500, width=1200,
        )
        return fig

    def plot_retrieval_comparison(self, benchmark_results):
        """Bar chart comparing retriever latencies (p50/p95)."""
        retrievers = []
        p50_values = []
        p95_values = []

        for name, metrics in benchmark_results.items():
            if "error" in metrics or "latency" not in metrics:
                continue
            retrievers.append(name)
            p50_values.append(metrics["latency"]["p50_ms"])
            p95_values.append(metrics["latency"]["p95_ms"])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="p50 Latency (ms)",
            x=retrievers, y=p50_values,
            marker_color="#636EFA",
        ))
        fig.add_trace(go.Bar(
            name="p95 Latency (ms)",
            x=retrievers, y=p95_values,
            marker_color="#EF553B",
        ))

        fig.update_layout(
            title="Retriever Latency Comparison",
            xaxis_title="Retriever",
            yaxis_title="Latency (ms)",
            barmode="group",
            template="plotly_dark",
            height=450, width=800,
        )
        return fig

    def plot_cluster_comparison(self, comparison_results):
        """Bar chart comparing KMeans vs BERTopic quality metrics."""
        methods = []
        metrics_data = {"Silhouette": [], "Davies-Bouldin": []}

        for method_key in ["kmeans", "bertopic"]:
            if method_key not in comparison_results:
                continue
            m = comparison_results[method_key]
            methods.append(m["method"])
            metrics_data["Silhouette"].append(m.get("silhouette") or 0)
            metrics_data["Davies-Bouldin"].append(m.get("davies_bouldin") or 0)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Silhouette Score (higher = better)",
                            "Davies-Bouldin Index (lower = better)"],
        )

        fig.add_trace(
            go.Bar(x=methods, y=metrics_data["Silhouette"],
                   marker_color=["#636EFA", "#00CC96"],
                   showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=methods, y=metrics_data["Davies-Bouldin"],
                   marker_color=["#636EFA", "#00CC96"],
                   showlegend=False),
            row=1, col=2,
        )

        fig.update_layout(
            title="Clustering Quality: KMeans vs BERTopic",
            template="plotly_dark",
            height=400, width=900,
        )
        return fig
