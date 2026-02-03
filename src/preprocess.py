import pandas as pd
import re


def clean_text(text):
    """Clean and normalize complaint text."""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
    
    return text.strip()


def load_and_preprocess(input_path, output_path=None, sample_size=15000):
    """
    Load complaints data, filter, clean, and optionally save.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save processed CSV (optional)
        sample_size: Number of rows to sample (default 15000)
    
    Returns:
        DataFrame with processed complaints
    """
    # Only load columns we need - much faster for large CSV
    required_columns = ['Consumer complaint narrative', 'Product', 'Issue', 'Date received']
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, usecols=required_columns, low_memory=True)
    print(f"Total rows loaded: {len(df)}")
    
    # Filter rows with complaint narrative
    df_filtered = df[df['Consumer complaint narrative'].notna()].copy()
    print(f"Rows with complaint narrative: {len(df_filtered)}")
    
    # Sample data
    sample_size = min(sample_size, len(df_filtered))
    df_sample = df_filtered.sample(n=sample_size, random_state=42)
    print(f"Sampled {sample_size} rows")
    
    # Keep required columns and rename
    df_clean = df_sample[['Consumer complaint narrative', 'Product', 'Issue', 'Date received']].copy()
    df_clean.columns = ['complaint_text', 'product', 'issue', 'date']
    
    # Clean text
    print("Cleaning text...")
    df_clean['clean_text'] = df_clean['complaint_text'].apply(clean_text)
    
    # Handle missing values
    df_clean['product'] = df_clean['product'].fillna('Unknown')
    df_clean['issue'] = df_clean['issue'].fillna('Unknown')
    
    # Convert date
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Remove rows with very short cleaned text
    df_clean = df_clean[df_clean['clean_text'].str.len() > 10].copy()
    
    print(f"Final dataset: {len(df_clean)} rows")
    
    # Save if output path provided
    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
    
    return df_clean


if __name__ == "__main__":
    # Example usage
    input_file = "../data/raw/complaints.csv"
    output_file = "../data/processed/processed_complaints.csv"
    
    df = load_and_preprocess(input_file, output_file)
    print(f"\nProcessed {len(df)} complaints")
    print(f"Columns: {list(df.columns)}")
