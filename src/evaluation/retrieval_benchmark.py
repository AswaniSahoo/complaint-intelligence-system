"""
Retrieval Benchmark - compare retrieval strategies head-to-head.

Runs a fixed set of test queries across every retriever and measures:
    - Latency: p50, p95, p99 response times
    - Retrieval quality via latency comparison

Outputs JSON results that the dashboard can consume for radar charts
and comparison tables.
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pathlib import Path

from src.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# Test queries covering different complaint domains
DEFAULT_QUERIES = [
    "credit card billing dispute charges",
    "mortgage loan modification denied",
    "debt collector calling me about a debt I don't owe",
    "unauthorized transactions on my bank account",
    "credit report errors wrong information",
    "student loan payment not applied correctly",
    "identity theft someone opened account in my name",
    "late fees charged incorrectly",
    "auto loan interest rate too high",
    "checking account closed without notice",
    "payday loan predatory lending practices",
    "foreclosure process unfair",
    "wire transfer lost money",
    "credit card rewards not received",
    "bank account overdraft fees excessive",
    "medical debt on credit report",
    "unable to contact customer service",
    "prepaid card balance missing",
    "home equity line of credit problems",
    "money transfer to wrong account",
]


class RetrievalBenchmark:
    """Benchmark multiple retrieval strategies on the same query set."""

    def __init__(self, queries=None):
        self.queries = queries or DEFAULT_QUERIES
        self.results = {}

    def benchmark_retriever(self, retriever: BaseRetriever, k: int = 5,
                            n_runs: int = 1):
        """Benchmark a single retriever and calculate latency metrics."""
        name = retriever.name
        print(f"\nBenchmarking retriever: {name}")
        print(f"  Queries: {len(self.queries)}, k={k}, runs={n_runs}")

        latencies = []
        result_counts = []
        all_results = {}

        for query in self.queries:
            query_latencies = []
            for _ in range(n_runs):
                results, latency_ms = retriever.retrieve_with_timing(query, k=k)
                query_latencies.append(latency_ms)

            avg_latency = np.mean(query_latencies)
            latencies.append(avg_latency)
            result_counts.append(len(results))

            all_results[query] = {
                "latency_ms": round(avg_latency, 2),
                "num_results": len(results),
                "top_scores": [round(r.score, 4) for r in results[:3]],
            }

        latency_array = np.array(latencies)

        metrics = {
            "retriever": name,
            "num_queries": len(self.queries),
            "k": k,
            "latency": {
                "mean_ms": round(float(np.mean(latency_array)), 2),
                "p50_ms": round(float(np.percentile(latency_array, 50)), 2),
                "p95_ms": round(float(np.percentile(latency_array, 95)), 2),
                "p99_ms": round(float(np.percentile(latency_array, 99)), 2),
                "min_ms": round(float(np.min(latency_array)), 2),
                "max_ms": round(float(np.max(latency_array)), 2),
            },
            "avg_results_returned": round(float(np.mean(result_counts)), 1),
            "per_query": all_results,
        }

        print(f"  Latency: p50={metrics['latency']['p50_ms']:.1f}ms, "
              f"p95={metrics['latency']['p95_ms']:.1f}ms, "
              f"p99={metrics['latency']['p99_ms']:.1f}ms")
        print(f"  Avg results: {metrics['avg_results_returned']}")

        self.results[name] = metrics
        return metrics

    def benchmark_all(self, retrievers: Dict[str, BaseRetriever],
                      k: int = 5, n_runs: int = 1):
        """Benchmark all provided retrievers on the query set."""
        print("=" * 60)
        print("RETRIEVAL BENCHMARK")
        print("=" * 60)
        print(f"Retrievers: {list(retrievers.keys())}")
        print(f"Queries: {len(self.queries)}")

        for name, retriever in retrievers.items():
            try:
                self.benchmark_retriever(retriever, k=k, n_runs=n_runs)
            except Exception as e:
                logger.error("Failed to benchmark %s: %s", name, e)
                self.results[name] = {"retriever": name, "error": str(e)}

        return self.results

    def print_summary(self):
        """Print a comparison table."""
        print("\n" + "=" * 60)
        print("RETRIEVAL BENCHMARK SUMMARY")
        print("=" * 60)

        header = (f"{'Retriever':<25} {'p50 (ms)':>10} {'p95 (ms)':>10} "
                  f"{'p99 (ms)':>10} {'Avg Results':>12}")
        print(header)
        print("-" * len(header))

        for name, metrics in self.results.items():
            if "error" in metrics:
                print(f"{name:<25} {'ERROR':>10}")
                continue

            lat = metrics["latency"]
            print(
                f"{name:<25} {lat['p50_ms']:>10.1f} {lat['p95_ms']:>10.1f} "
                f"{lat['p99_ms']:>10.1f} {metrics['avg_results_returned']:>12.1f}"
            )

    def save_results(self, output_path):
        """Save benchmark results as JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nBenchmark results saved to {output_path}")

    @staticmethod
    def load_results(input_path):
        """Load benchmark results from JSON."""
        with open(input_path, "r") as f:
            return json.load(f)
