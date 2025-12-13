"""
Complete pipeline script to process complaints end-to-end.
Run this after setting up your environment and API keys.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import load_and_preprocess
from src.embeddings import generate_embeddings
from src.clustering import cluster_complaints
from src.llm_utils import summarize_complaints


def main():
    """Run the complete pipeline."""
    
    print("=" * 60)
    print("CUSTOMER COMPLAINT INTELLIGENCE SYSTEM - PIPELINE")
    print("=" * 60)
    
    # Paths
    raw_data = "data/raw/complaints.csv"
    processed_data = "data/processed/processed_complaints.csv"
    embeddings_path = "data/processed/embeddings.npy"
    
    # Step 1: Preprocess
    print("\n[1/4] PREPROCESSING DATA...")
    print("-" * 60)
    df = load_and_preprocess(raw_data, processed_data, sample_size=15000)
    print(f"✓ Processed {len(df)} complaints")
    
    # Step 2: Generate embeddings
    print("\n[2/4] GENERATING EMBEDDINGS...")
    print("-" * 60)
    embeddings = generate_embeddings(df, text_column='clean_text', output_path=embeddings_path)
    print(f"✓ Generated embeddings: {embeddings.shape}")
    
    # Step 3: Cluster
    print("\n[3/4] CLUSTERING COMPLAINTS...")
    print("-" * 60)
    df, cluster_keywords = cluster_complaints(df, embeddings, n_clusters=6)
    
    print("\nCluster Keywords:")
    for cluster_id, keywords in cluster_keywords.items():
        print(f"  Cluster {cluster_id}: {', '.join(keywords[:5])}")
    
    # Save with clusters
    df.to_csv(processed_data, index=False)
    print(f"✓ Saved clustered data to {processed_data}")
    
    # Step 4: LLM Summarization (optional - can be slow)
    run_llm = input("\n[4/4] Run LLM summarization? This may take time and requires API keys (y/n): ").lower()
    
    if run_llm == 'y':
        print("\nGENERATING LLM SUMMARIES...")
        print("-" * 60)
        
        # Ask which provider
        provider = input("Choose provider (gemini/groq): ").lower()
        if provider not in ['gemini', 'groq']:
            provider = 'gemini'
        
        # Process a smaller sample first for testing
        sample_size = int(input("How many complaints to process? (recommended: 100-1000): ") or "500")
        df_sample = df.head(sample_size)
        
        df_sample = summarize_complaints(df_sample, provider=provider, batch_size=10)
        
        # Merge back to main df
        for col in ['llm_summary', 'llm_category', 'llm_urgency']:
            if col in df_sample.columns:
                df[col] = None
                df.loc[df_sample.index, col] = df_sample[col]
        
        df.to_csv(processed_data, index=False)
        print(f"✓ Saved LLM results to {processed_data}")
    else:
        print("⊘ Skipping LLM summarization")
    
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
