"""
Complete pipeline script to process complaints end-to-end.
Run this after setting up your environment and API keys.

Usage:
    python run_pipeline.py                          # Full pipeline (no LLM)
    python run_pipeline.py --sample-size 5000       # Custom sample size
    python run_pipeline.py --with-llm               # Include LLM summarization
    python run_pipeline.py --with-llm --provider groq --llm-sample 500
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent

from src.preprocess import load_and_preprocess
from src.embeddings import generate_embeddings
from src.clustering import cluster_complaints
from src.llm_utils import summarize_complaints


def parse_args():
    """Parse command-line arguments."""
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
        "--raw-data", type=str, default=None,
        help="Path to raw CSV (default: data/raw/complaints.csv)"
    )
    return parser.parse_args()


def main():
    """Run the complete pipeline."""
    args = parse_args()

    # Paths
    raw_data = args.raw_data or str(PROJECT_ROOT / "data" / "raw" / "complaints.csv")
    processed_data = str(PROJECT_ROOT / "data" / "processed" / "processed_complaints.csv")
    embeddings_path = str(PROJECT_ROOT / "data" / "processed" / "embeddings.npy")

    print("=" * 60)
    print("CUSTOMER COMPLAINT INTELLIGENCE SYSTEM - PIPELINE")
    print("=" * 60)
    print(f"\nConfig: sample_size={args.sample_size}, clusters={args.n_clusters}, "
          f"llm={'yes' if args.with_llm else 'no'}")

    # Step 1: Preprocess
    print("\n[1/4] PREPROCESSING DATA...")
    print("-" * 60)
    df = load_and_preprocess(raw_data, processed_data, sample_size=args.sample_size)
    print(f"✓ Processed {len(df)} complaints")

    # Step 2: Generate embeddings
    print("\n[2/4] GENERATING EMBEDDINGS...")
    print("-" * 60)
    embeddings = generate_embeddings(df, text_column='clean_text', output_path=embeddings_path)
    print(f"✓ Generated embeddings: {embeddings.shape}")

    # Step 3: Cluster
    print("\n[3/4] CLUSTERING COMPLAINTS...")
    print("-" * 60)
    df, cluster_keywords = cluster_complaints(df, embeddings, n_clusters=args.n_clusters)

    print("\nCluster Keywords:")
    for cluster_id, keywords in cluster_keywords.items():
        print(f"  Cluster {cluster_id}: {', '.join(keywords[:5])}")

    # Save with clusters
    df.to_csv(processed_data, index=False)
    print(f"✓ Saved clustered data to {processed_data}")

    # Step 4: LLM Summarization (optional)
    if args.with_llm:
        print("\n[4/4] GENERATING LLM SUMMARIES...")
        print("-" * 60)
        print(f"Provider: {args.provider}, Sample: {args.llm_sample}")

        llm_sample = min(args.llm_sample, len(df))
        df_sample = df.head(llm_sample)

        df_sample = summarize_complaints(df_sample, provider=args.provider, batch_size=10)

        # Merge back to main df
        for col in ['llm_summary', 'llm_category', 'llm_urgency']:
            if col in df_sample.columns:
                df[col] = None
                df.loc[df_sample.index, col] = df_sample[col]

        df.to_csv(processed_data, index=False)
        print(f"✓ Saved LLM results to {processed_data}")
    else:
        print("\n[4/4] Skipping LLM summarization (use --with-llm to enable)")

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nProcessed files:")
    print(f"  - {processed_data}")
    print(f"  - {embeddings_path}")
    print(f"\nDataset stats:")
    print(f"  - Total complaints: {len(df)}")
    print(f"  - Clusters: {df['cluster'].nunique()}")
    if 'llm_summary' in df.columns:
        print(f"  - LLM summaries: {df['llm_summary'].notna().sum()}")

    print("\n" + "=" * 60)
    print("Next step: Run the dashboard")
    print("  streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
