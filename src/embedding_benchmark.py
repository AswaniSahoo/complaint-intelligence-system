"""
Embedding benchmark - compare models on throughput, similarity, and cluster separation.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pathlib import Path

from src.embeddings import EmbeddingRegistry, EMBEDDING_MODELS


class EmbeddingBenchmark:
    """Run head-to-head comparison across multiple embedding models."""

    def __init__(self, model_keys=None):
        self.model_keys = model_keys or list(EMBEDDING_MODELS.keys())
        self.registry = EmbeddingRegistry()
        self.results = {}

    def run(self, texts, cluster_labels=None, batch_size=64):
        """Run the full benchmark. Returns dict of results per model."""
        print(f"Embedding benchmark: {len(texts)} texts, models={self.model_keys}")

        for key in self.model_keys:
            self.registry.load_model(key)

        all_embeddings = {}

        for key in self.model_keys:
            print(f"\nBenchmarking: {key} ({EMBEDDING_MODELS[key]['name']})")


            start = time.perf_counter()
            embeddings = self.registry.encode(
                key, texts, batch_size=batch_size, show_progress=True
            )
            elapsed = time.perf_counter() - start

            all_embeddings[key] = embeddings

            throughput = len(texts) / elapsed
            memory_mb = embeddings.nbytes / (1024 * 1024)


            sim_stats = self._cosine_similarity_stats(embeddings, n_pairs=5000)


            cluster_stats = {}
            if cluster_labels is not None:
                cluster_stats = self._cluster_similarity(
                    embeddings, cluster_labels
                )

            self.results[key] = {
                "model_name": EMBEDDING_MODELS[key]["name"],
                "embedding_dim": int(embeddings.shape[1]),
                "num_texts": len(texts),
                "encoding_time_sec": round(elapsed, 2),
                "throughput_texts_per_sec": round(throughput, 1),
                "memory_mb": round(memory_mb, 2),
                "cosine_similarity": sim_stats,
                "cluster_separation": cluster_stats,
            }

            print(f"\n  Throughput: {throughput:.1f} texts/sec")
            print(f"  Encoding time: {elapsed:.2f}s")
            print(f"  Memory: {memory_mb:.2f} MB")
            print(f"  Cosine sim (mean): {sim_stats['mean']:.4f}")

        # Compare nearest neighbors across models
        if len(all_embeddings) >= 2:
            self.results["cross_model"] = self._cross_model_comparison(
                all_embeddings, texts
            )

        return self.results

    def _cosine_similarity_stats(self, embeddings, n_pairs=5000):
        """Compute cosine similarity statistics over random pairs."""
        n = len(embeddings)
        n_pairs = min(n_pairs, n * (n - 1) // 2)

        # Normalize for cosine sim
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-10)


        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, n, size=n_pairs)
        idx_b = rng.integers(0, n, size=n_pairs)
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]

        sims = np.sum(normed[idx_a] * normed[idx_b], axis=1)

        return {
            "mean": round(float(np.mean(sims)), 4),
            "std": round(float(np.std(sims)), 4),
            "min": round(float(np.min(sims)), 4),
            "max": round(float(np.max(sims)), 4),
            "p25": round(float(np.percentile(sims, 25)), 4),
            "p50": round(float(np.percentile(sims, 50)), 4),
            "p75": round(float(np.percentile(sims, 75)), 4),
        }

    def _cluster_similarity(self, embeddings, labels):
        """Compute intra-cluster vs inter-cluster average cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-10)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]

        if len(unique_labels) < 2:
            return {}

        rng = np.random.default_rng(42)
        intra_sims = []
        inter_sims = []

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_vecs = normed[cluster_mask]
            other_vecs = normed[~cluster_mask]

            if len(cluster_vecs) < 2 or len(other_vecs) < 1:
                continue

            # Sample intra-cluster pairs
            n_sample = min(200, len(cluster_vecs))
            idx = rng.choice(len(cluster_vecs), size=(n_sample, 2), replace=True)
            mask = idx[:, 0] != idx[:, 1]
            pairs = idx[mask]
            if len(pairs) > 0:
                sims = np.sum(
                    cluster_vecs[pairs[:, 0]] * cluster_vecs[pairs[:, 1]], axis=1
                )
                intra_sims.extend(sims.tolist())

            # Sample inter-cluster pairs
            n_sample = min(200, len(other_vecs))
            c_idx = rng.choice(len(cluster_vecs), size=n_sample, replace=True)
            o_idx = rng.choice(len(other_vecs), size=n_sample, replace=True)
            sims = np.sum(cluster_vecs[c_idx] * other_vecs[o_idx], axis=1)
            inter_sims.extend(sims.tolist())

        if not intra_sims or not inter_sims:
            return {}

        separation = float(np.mean(intra_sims)) - float(np.mean(inter_sims))

        return {
            "intra_cluster_sim": round(float(np.mean(intra_sims)), 4),
            "inter_cluster_sim": round(float(np.mean(inter_sims)), 4),
            "separation": round(separation, 4),
            "n_clusters": int(len(unique_labels)),
        }

    def _cross_model_comparison(self, all_embeddings, texts):
        """Compare how two models agree/disagree on similarity rankings."""
        keys = list(all_embeddings.keys())
        if len(keys) < 2:
            return {}

        key_a, key_b = keys[0], keys[1]
        emb_a = all_embeddings[key_a]
        emb_b = all_embeddings[key_b]


        norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-10)
        norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-10)


        rng = np.random.default_rng(42)
        n_queries = min(100, len(texts))
        query_indices = rng.choice(len(texts), size=n_queries, replace=False)

        overlaps = []
        for qi in query_indices:
            sims_a = norm_a[qi] @ norm_a.T
            sims_b = norm_b[qi] @ norm_b.T

            top_a = set(np.argsort(sims_a)[-11:-1])  # exclude self
            top_b = set(np.argsort(sims_b)[-11:-1])

            overlap = len(top_a & top_b) / 10.0
            overlaps.append(overlap)

        return {
            "model_a": key_a,
            "model_b": key_b,
            "top10_overlap_mean": round(float(np.mean(overlaps)), 4),
            "top10_overlap_std": round(float(np.std(overlaps)), 4),
            "interpretation": (
                "Higher overlap means models agree on which texts are similar. "
                "Low overlap (<0.3) suggests they capture different semantic aspects."
            ),
        }

    def save_results(self, output_path):
        """Save benchmark results as JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nBenchmark results saved to {output_path}")

    def print_summary(self):
        """Print a comparison table of all models."""
        print(f"\nBenchmark summary:")

        header = f"{'Model':<12} {'Dim':>5} {'Speed (t/s)':>12} {'Mem (MB)':>10} {'Sim Mean':>10} {'Separation':>12}"
        print(header)
        print("-" * len(header))

        for key in self.model_keys:
            if key not in self.results:
                continue
            r = self.results[key]
            sep = r.get("cluster_separation", {}).get("separation", "N/A")
            sep_str = f"{sep:.4f}" if isinstance(sep, float) else sep
            print(
                f"{key:<12} {r['embedding_dim']:>5} "
                f"{r['throughput_texts_per_sec']:>12.1f} "
                f"{r['memory_mb']:>10.2f} "
                f"{r['cosine_similarity']['mean']:>10.4f} "
                f"{sep_str:>12}"
            )

        if "cross_model" in self.results:
            cm = self.results["cross_model"]
            print(f"\nCross-model top-10 neighbor overlap: "
                  f"{cm['top10_overlap_mean']:.1%} ± {cm['top10_overlap_std']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run embedding model benchmark")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Number of texts to benchmark (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size (default: 64)")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_path = PROJECT_ROOT / "data" / "processed" / "processed_complaints.csv"
    results_dir = PROJECT_ROOT / "data" / "results"

    df = pd.read_csv(data_path)
    df_sample = df.head(args.sample)

    texts = df_sample["clean_text"].tolist()
    cluster_labels = df_sample["cluster"].values if "cluster" in df_sample.columns else None

    benchmark = EmbeddingBenchmark(model_keys=["minilm", "bge"])
    benchmark.run(texts, cluster_labels=cluster_labels, batch_size=args.batch_size)
    benchmark.print_summary()
    benchmark.save_results(str(results_dir / "embedding_benchmark.json"))
