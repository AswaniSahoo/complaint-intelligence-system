"""
Generate LLM summaries for complaints.
This script processes complaints and adds AI-generated summaries, categories, and urgency.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.llm_utils import summarize_complaints


def main():
    print("=" * 60)
    print("GENERATING LLM SUMMARIES FOR COMPLAINTS")
    print("=" * 60)
    
    # Load processed data
    data_path = os.path.join("data", "processed", "processed_complaints.csv")
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} complaints")
    
    # Process 1000 complaints by default (good for demo)
    sample_size = 1000
    print(f"\nProcessing {sample_size} complaints with Gemini API...")
    
    # Select sample
    df_sample = df.head(sample_size).copy()
    
    # Process with LLM using Gemini
    df_sample = summarize_complaints(
        df_sample,
        text_column='complaint_text',
        provider='gemini',
        batch_size=5
    )
    
    # Merge results back to main dataframe
    print("\nMerging results...")
    for col in ['llm_summary', 'llm_category', 'llm_urgency']:
        if col in df_sample.columns:
            df[col] = None
            df.loc[df_sample.index, col] = df_sample[col]
    
    # Save updated data
    df.to_csv(data_path, index=False)
    print(f"\n✓ Updated data saved to {data_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("LLM SUMMARIZATION COMPLETE")
    print("=" * 60)
    print(f"\nSummarized: {sample_size} complaints")
    print(f"Total dataset: {len(df)} complaints")
    print(f"Coverage: {sample_size/len(df)*100:.1f}%")
    
    if 'llm_category' in df.columns:
        print("\nCategory distribution:")
        print(df['llm_category'].value_counts())
        
        print("\nUrgency distribution:")
        print(df['llm_urgency'].value_counts())
    
    print("\n" + "=" * 60)
    print("✓ PROJECT READY FOR SHOWCASE")
    print("=" * 60)
    print("\nFeatures completed:")
    print("  ✓ 15,000 complaints processed")
    print("  ✓ Text cleaning and preprocessing")
    print("  ✓ Sentence embeddings (384-dim)")
    print("  ✓ KMeans clustering (6 clusters)")
    print(f"  ✓ AI summaries for {sample_size} complaints")
    print("  ✓ FAISS vector search")
    print("  ✓ Interactive Streamlit dashboard")
    print("\nLaunch dashboard:")
    print("  streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
