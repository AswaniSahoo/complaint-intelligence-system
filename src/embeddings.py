"""
Embedding generation module for complaint text using sentence-transformers.
Supports GPU acceleration when CUDA is available.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import torch


class ComplaintEmbedder:
    """Generate embeddings for complaint text using sentence-transformers."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        # Auto-detect GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading embedding model: {model_name}")
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts, batch_size=32, show_progress=True):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def save_embeddings(self, embeddings, output_path):
        """Save embeddings to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Embeddings saved to {output_path}")
    
    def load_embeddings(self, input_path):
        """Load embeddings from disk."""
        embeddings = np.load(input_path)
        print(f"Loaded embeddings: {embeddings.shape}")
        return embeddings


def generate_embeddings(df, text_column='clean_text', output_path=None):
    """
    Generate embeddings for complaints dataframe.
    
    Args:
        df: DataFrame with complaint text
        text_column: Column name containing text to embed
        output_path: Path to save embeddings (optional)
    
    Returns:
        numpy array of embeddings
    """
    embedder = ComplaintEmbedder()
    embeddings = embedder.encode(df[text_column])
    
    if output_path:
        embedder.save_embeddings(embeddings, output_path)
    
    return embeddings


if __name__ == "__main__":
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "processed_complaints.csv")
    output_path = os.path.join(base_dir, "data", "processed", "embeddings.npy")
    
    df = pd.read_csv(data_path)
    embeddings = generate_embeddings(
        df,
        text_column='clean_text',
        output_path=output_path
    )
    print(f"Generated embeddings shape: {embeddings.shape}")
