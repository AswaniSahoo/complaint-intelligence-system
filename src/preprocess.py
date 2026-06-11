import pandas as pd
import re


def clean_text(text):
    """Clean and normalize complaint text."""
    try:
        if pd.isna(text):
            return ""
    except (ValueError, TypeError):
        pass

    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
    return text.strip()


def load_and_preprocess(input_path, output_path=None, sample_size=15000):
    """Load complaints CSV, filter for narratives, clean text, and save."""
    required_columns = ['Consumer complaint narrative', 'Product', 'Issue', 'Date received']

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, usecols=required_columns, low_memory=True)
    print(f"Loaded {len(df)} rows")

    df_filtered = df[df['Consumer complaint narrative'].notna()].copy()
    print(f"Rows with narratives: {len(df_filtered)}")

    sample_size = min(sample_size, len(df_filtered))
    df_sample = df_filtered.sample(n=sample_size, random_state=42)

    df_clean = df_sample[required_columns].copy()
    df_clean.columns = ['complaint_text', 'product', 'issue', 'date']

    print("Cleaning text...")
    df_clean['clean_text'] = df_clean['complaint_text'].apply(clean_text)

    df_clean['product'] = df_clean['product'].fillna('Unknown')
    df_clean['issue'] = df_clean['issue'].fillna('Unknown')
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')

    # Drop rows where cleaned text is too short to be useful
    df_clean = df_clean[df_clean['clean_text'].str.len() > 10].copy()
    print(f"Final: {len(df_clean)} rows")

    if output_path:
        df_clean.to_csv(output_path, index=False)

    return df_clean


if __name__ == "__main__":
    df = load_and_preprocess("../data/raw/complaints.csv",
                             "../data/processed/processed_complaints.csv")
    print(f"Columns: {list(df.columns)}")
