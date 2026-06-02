"""
Embedding generation module for complaint text using sentence-transformers.
Supports GPU acceleration when CUDA is available.

Models:
    - all-MiniLM-L6-v2 (384d): Fast, lightweight baseline (2022-era)
    - BAAI/bge-base-en-v1.5 (768d): SOTA MTEB performer (2024-era)

Provides multi-model comparison for benchmarking old vs new embeddings.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import time
import torch


# -- Model Registry -----------------------------------------------------------

EMBEDDING_MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Lightweight baseline (2022), fast encoding",
        "query_prefix": "",
    },
    "bge": {
        "name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "description": "SOTA MTEB performer (2024), high accuracy",
        "query_prefix": "Represent this sentence: ",
    },
}

DEFAULT_MODEL = "minilm"


# -- Core Embedder ------------------------------------------------------------

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


# -- Multi-Model Registry -----------------------------------------------------

class EmbeddingRegistry:
    """Manage multiple embedding models for side-by-side comparison.

    Example::

        registry = EmbeddingRegistry()
        registry.load_model("minilm")
        registry.load_model("bge")

        results = registry.encode_all(texts)
        # results = {"minilm": np.array(...), "bge": np.array(...)}
    """

    def __init__(self):
        self.models = {}   # key -> ComplaintEmbedder
        self.configs = {}  # key -> model config dict

    def load_model(self, key):
        """
        Load a model by its registry key.

        Args:
            key: One of the keys in EMBEDDING_MODELS (e.g. "minilm", "bge")
        """
        if key in self.models:
            print(f"Model '{key}' already loaded, skipping.")
            return

        if key not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown model key '{key}'. "
                f"Available: {list(EMBEDDING_MODELS.keys())}"
            )

        config = EMBEDDING_MODELS[key]
        self.configs[key] = config
        self.models[key] = ComplaintEmbedder(model_name=config["name"])

    def encode(self, key, texts, batch_size=32, show_progress=True):
        """
        Encode texts with a specific model.

        Args:
            key: Model registry key.
            texts: List of text strings.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            numpy array of embeddings
        """
        if key not in self.models:
            raise ValueError(f"Model '{key}' not loaded. Call load_model('{key}') first.")

        config = self.configs[key]
        prefix = config.get("query_prefix", "")

        if prefix and isinstance(texts, list):
            texts = [prefix + t for t in texts]
        elif prefix and isinstance(texts, pd.Series):
            texts = (prefix + texts).tolist()

        return self.models[key].encode(texts, batch_size=batch_size, show_progress=show_progress)

    def encode_all(self, texts, batch_size=32, show_progress=True):
        """
        Encode texts with ALL loaded models and return a dict.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            dict mapping model key to numpy array of embeddings
        """
        results = {}
        for key in self.models:
            print(f"\n--- Encoding with '{key}' ({self.configs[key]['name']}) ---")
            results[key] = self.encode(
                key, texts, batch_size=batch_size, show_progress=show_progress
            )
        return results

    def get_embedder(self, key):
        """Get the ComplaintEmbedder instance for a model key."""
        if key not in self.models:
            raise ValueError(f"Model '{key}' not loaded.")
        return self.models[key]

    def get_dim(self, key):
        """Get embedding dimension for a model key."""
        return EMBEDDING_MODELS[key]["dim"]

    @property
    def loaded_models(self):
        """List of currently loaded model keys."""
        return list(self.models.keys())


# -- Convenience functions (backward compatible) -------------------------------

def generate_embeddings(df, text_column='clean_text', output_path=None,
                        model_key=None):
    """
    Generate embeddings for complaints dataframe.
    
    Args:
        df: DataFrame with complaint text
        text_column: Column name containing text to embed
        output_path: Path to save embeddings (optional)
        model_key: Registry key (e.g. "minilm", "bge"). If None, uses
                   DEFAULT_MODEL for backward compatibility.
    
    Returns:
        numpy array of embeddings
    """
    if model_key and model_key in EMBEDDING_MODELS:
        config = EMBEDDING_MODELS[model_key]
        embedder = ComplaintEmbedder(model_name=config["name"])
        prefix = config.get("query_prefix", "")
        texts = df[text_column].tolist()
        if prefix:
            texts = [prefix + t for t in texts]
        embeddings = embedder.encode(texts)
    else:
        embedder = ComplaintEmbedder()
        embeddings = embedder.encode(df[text_column])
    
    if output_path:
        embedder.save_embeddings(embeddings, output_path)
    
    return embeddings


def generate_all_embeddings(df, text_column='clean_text', output_dir=None,
                            model_keys=None):
    """
    Generate embeddings from multiple models and save each separately.

    Args:
        df: DataFrame with complaint text.
        text_column: Column name containing text to embed.
        output_dir: Directory to save embedding files (optional).
        model_keys: List of registry keys. Defaults to all available.

    Returns:
        dict mapping model key to numpy array of embeddings
    """
    if model_keys is None:
        model_keys = list(EMBEDDING_MODELS.keys())

    registry = EmbeddingRegistry()
    for key in model_keys:
        registry.load_model(key)

    results = registry.encode_all(df[text_column].tolist())

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for key, embeddings in results.items():
            path = os.path.join(output_dir, f"embeddings_{key}.npy")
            np.save(path, embeddings)
            print(f"Saved {key} embeddings to {path}")

    return results


if __name__ == "__main__":
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "processed_complaints.csv")
    output_dir = os.path.join(base_dir, "data", "processed")
    
    df = pd.read_csv(data_path)

    # Generate embeddings with both models
    all_embeddings = generate_all_embeddings(
        df,
        text_column='clean_text',
        output_dir=output_dir,
        model_keys=["minilm", "bge"]
    )

    for key, emb in all_embeddings.items():
        print(f"{key}: {emb.shape}")
