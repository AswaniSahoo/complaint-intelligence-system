"""Main pipeline script. Processes complaints end-to-end."""


import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

from src.preprocess import load_and_preprocess
from src.embeddings import (
    generate_embeddings, generate_all_embeddings, EMBEDDING_MODELS,
)
from src.clustering import (
    cluster_complaints, KMeansClusterer, BERTopicClusterer, ClusterComparison,
)
from src.llm_utils import summarize_complaints


def parse_args():
    parser = argparse.ArgumentParser(
        description="Customer Complaint Intelligence System - Data Processing Pipeline"
    )
    parser.add_argument(
        "--sample-size", type=int, default=15000,
        help="Number of complaints to sample from raw data (default: 15000)"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=6,
        help="Number of KMeans clusters (default: 6)"
    )
    parser.add_argument(
        "--model", type=str, default="minilm",
        choices=["minilm", "bge", "both"],
        help="Embedding model to use (default: minilm)"
    )
    parser.add_argument(
        "--clustering", type=str, default="kmeans",
        choices=["kmeans", "bertopic", "both"],
        help="Clustering method (default: kmeans)"
    )
    parser.add_argument(
        "--with-llm", action="store_true",
        help="Run LLM summarization step (requires API keys)"
    )
    parser.add_argument(
        "--provider", type=str, default="gemini",
        choices=["gemini", "groq", "together"],
        help="LLM provider for summarization (default: gemini)"
    )
    parser.add_argument(
        "--llm-sample", type=int, default=500,
        help="Number of complaints to summarize with LLM (default: 500)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run retrieval benchmarks after pipeline"
    )
    parser.add_argument(
        "--raw-data", type=str, default=None,
        help="Path to raw CSV (default: data/raw/complaints.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_data = args.raw_data or str(PROJECT_ROOT / "data" / "raw" / "complaints.csv")
    processed_data = str(PROJECT_ROOT / "data" / "processed" / "processed_complaints.csv")
    processed_dir = str(PROJECT_ROOT / "data" / "processed")
    results_dir = str(PROJECT_ROOT / "data" / "results")

    print(f"Pipeline: {args.sample_size} complaints, model={args.model}, "
          f"clustering={args.clustering}")

    print("\n[1/6] Preprocessing...")
    df = load_and_preprocess(raw_data, processed_data, sample_size=args.sample_size)
    print(f"Processed {len(df)} complaints")

    print("\n[2/6] Generating embeddings...")

    if args.model == "both":
        all_embeddings = generate_all_embeddings(
            df, text_column='clean_text', output_dir=processed_dir,
            model_keys=["minilm", "bge"]
        )
        embeddings = all_embeddings["minilm"]
        for key, emb in all_embeddings.items():
            print(f"  {key}: {emb.shape}")
    else:
        model_key = args.model
        embeddings = generate_embeddings(
            df, text_column='clean_text',
            output_path=os.path.join(processed_dir, f"embeddings_{model_key}.npy"),
            model_key=model_key
        )
        np.save(os.path.join(processed_dir, "embeddings.npy"), embeddings)
        print(f"{model_key} embeddings: {embeddings.shape}")

    print("\n[3/6] Clustering...")

    cluster_comparison = None

    if args.clustering in ("kmeans", "both"):
        print("\n--- KMeans Clustering ---")
        df, cluster_keywords = cluster_complaints(
            df, embeddings, n_clusters=args.n_clusters
        )
        kmeans_labels = df['cluster'].values.copy()

        print("\nKMeans Cluster Keywords:")
        for cluster_id, keywords in cluster_keywords.items():
            print(f"  Cluster {cluster_id}: {', '.join(keywords[:5])}")

    if args.clustering in ("bertopic", "both"):
        print("\n--- BERTopic ---")
        bt_clusterer = BERTopicClusterer(
            min_cluster_size=max(50, len(df) // 200),
            min_samples=10,
        )
        topic_labels = bt_clusterer.fit_predict(
            df['clean_text'].tolist(), embeddings
        )
        df['topic'] = topic_labels

        bt_model_path = os.path.join(processed_dir, "bertopic_model")
        bt_clusterer.save_model(bt_model_path)

        if args.clustering == "both":
            print("\n--- Cluster Quality Comparison ---")
            comparison = ClusterComparison.compare(
                embeddings, kmeans_labels, topic_labels
            )
            ClusterComparison.save_results(
                comparison,
                os.path.join(results_dir, "cluster_comparison.json")
            )
            cluster_comparison = comparison

        if args.clustering == "bertopic":
            df['cluster'] = topic_labels

    df.to_csv(processed_data, index=False)
    print(f"Saved to {processed_data}")

    if args.with_llm:
        print(f"\n[4/6] LLM summarization ({args.provider}, {args.llm_sample} samples)...")

        llm_sample = min(args.llm_sample, len(df))
        df_sample = df.head(llm_sample)

        df_sample = summarize_complaints(df_sample, provider=args.provider, batch_size=10)

        for col in ['llm_summary', 'llm_category', 'llm_urgency']:
            if col in df_sample.columns:
                df[col] = None
                df.loc[df_sample.index, col] = df_sample[col]

        df.to_csv(processed_data, index=False)
        print(f"Saved LLM results to {processed_data}")
    else:
        print("\n[4/6] Skipping LLM summarization (use --with-llm to enable)")

    if args.model == "both":
        print("\n[5/6] Embedding benchmark...")
        from src.embedding_benchmark import EmbeddingBenchmark

        emb_minilm = np.load(os.path.join(processed_dir, "embeddings_minilm.npy"))
        emb_bge = np.load(os.path.join(processed_dir, "embeddings_bge.npy"))

        benchmark = EmbeddingBenchmark(model_keys=["minilm", "bge"])
        sample_size = min(5000, len(df))
        texts_sample = df['clean_text'].head(sample_size).tolist()
        cluster_labels = df['cluster'].head(sample_size).values if 'cluster' in df.columns else None

        benchmark.run(texts_sample, cluster_labels=cluster_labels, batch_size=64)
        benchmark.print_summary()
        benchmark.save_results(os.path.join(results_dir, "embedding_benchmark.json"))
    else:
        print("\n[5/6] Skipping embedding benchmark (use --model both to enable)")

    if args.benchmark:
        print("\n[6/6] Retrieval benchmark...")
        from src.evaluation.retrieval_benchmark import RetrievalBenchmark
        from src.embeddings import ComplaintEmbedder
        from src.retrievers import (
            VectorRetriever, BM25Retriever, HybridRetriever, RerankedRetriever,
        )
        from src.retrievers.reranker import CrossEncoderReranker

        embedder = ComplaintEmbedder()

        vec = VectorRetriever(embedder=embedder, embedding_dim=embeddings.shape[1])
        vec.build_index(df, embeddings)

        bm25 = BM25Retriever()
        bm25.build_index(df)

        hybrid = HybridRetriever(vec, bm25)

        reranker = CrossEncoderReranker()
        reranked_hybrid = RerankedRetriever(hybrid, reranker, candidate_k=50)

        retrievers = {
            "vector": vec,
            "bm25": bm25,
            "hybrid": hybrid,
            "reranked_hybrid": reranked_hybrid,
        }

        bench = RetrievalBenchmark()
        bench.benchmark_all(retrievers, k=5)
        bench.print_summary()
        bench.save_results(os.path.join(results_dir, "retrieval_benchmark.json"))
    else:
        print("\n[6/6] Skipping retrieval benchmark (use --benchmark to enable)")

    print(f"\nDone. {len(df)} complaints, {df['cluster'].nunique()} clusters.")
    if 'topic' in df.columns:
        n_topics = df['topic'].nunique()
        n_outliers = (df['topic'] == -1).sum()
        print(f"BERTopic: {n_topics} topics, {n_outliers} outliers")
    print(f"\nRun the dashboard: streamlit run app/app.py")


if __name__ == "__main__":
    main()
