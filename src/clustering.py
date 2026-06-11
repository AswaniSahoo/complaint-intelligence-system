"""
Clustering module.

Two strategies:
    - KMeansClusterer: fixed-k centroid clustering
    - BERTopicClusterer: UMAP + HDBSCAN + c-TF-IDF topic modeling
    - ClusterComparison: quality metrics for both
"""

import logging
import json
import os

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import numpy as np
import pandas as pd
import pickle

logger = logging.getLogger(__name__)


class KMeansClusterer:
    """Cluster complaints using KMeans and extract keywords with TF-IDF."""
    
    def __init__(self, n_clusters=6, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_labels = None
    
    def fit_predict(self, embeddings):
        """Fit KMeans and return cluster labels."""
        print(f"Clustering {len(embeddings)} complaints into {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.cluster_labels = self.kmeans.fit_predict(embeddings)

        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} complaints ({count/len(self.cluster_labels)*100:.1f}%)")
        
        return self.cluster_labels
    
    def extract_keywords(self, df, text_column='clean_text', top_n=10):
        """Extract top TF-IDF keywords per cluster."""
        print(f"\nExtracting top {top_n} keywords per cluster...")
        cluster_keywords = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_texts = df[df['cluster'] == cluster_id][text_column].tolist()
            
            if len(cluster_texts) == 0:
                cluster_keywords[cluster_id] = []
                continue

            vectorizer = TfidfVectorizer(
                max_features=top_n,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            try:
                vectorizer.fit(cluster_texts)
                keywords = vectorizer.get_feature_names_out().tolist()
                cluster_keywords[cluster_id] = keywords
                print(f"  Cluster {cluster_id}: {', '.join(keywords[:5])}...")
            except Exception as e:
                logger.warning(f"Failed to extract keywords for cluster {cluster_id}: {e}")
                cluster_keywords[cluster_id] = []
        
        return cluster_keywords
    
    def save_model(self, output_path):
        """Save the KMeans model."""
        with open(output_path, 'wb') as f:
            pickle.dump(self.kmeans, f)
        print(f"Model saved to {output_path}")
    
    def load_model(self, input_path):
        """Load a saved KMeans model."""
        with open(input_path, 'rb') as f:
            self.kmeans = pickle.load(f)
        print(f"Model loaded from {input_path}")


class BERTopicClusterer:
    """Topic modeling using BERTopic (UMAP + HDBSCAN + c-TF-IDF)."""

    def __init__(self, min_cluster_size=50, min_samples=10,
                 umap_n_neighbors=15, umap_n_components=5,
                 ngram_range=(1, 2), random_state=42):
        """
        Args:
            min_cluster_size: HDBSCAN minimum cluster size.
            min_samples: HDBSCAN min_samples parameter.
            umap_n_neighbors: UMAP n_neighbors.
            umap_n_components: UMAP target dimensions.
            ngram_range: n-gram range for CountVectorizer / c-TF-IDF.
            random_state: Random seed for UMAP reproducibility.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.topic_model = None
        self.topics = None
        self.topic_info = None

    def fit_predict(self, texts, embeddings):
        """Fit BERTopic and return topic labels. -1 means outlier."""
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer

        print(f"Running BERTopic on {len(texts)} texts...")
        print(f"  UMAP: n_neighbors={self.umap_n_neighbors}, "
              f"n_components={self.umap_n_components}")
        print(f"  HDBSCAN: min_cluster_size={self.min_cluster_size}, "
              f"min_samples={self.min_samples}")

        umap_model = UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=self.umap_n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            prediction_data=True,
        )

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=self.ngram_range,
        )

        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            verbose=True,
        )

        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        self.topics, _ = self.topic_model.fit_transform(texts, embeddings)
        self.topic_info = self.topic_model.get_topic_info()

        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        n_outliers = sum(1 for t in self.topics if t == -1)
        outlier_pct = n_outliers / len(self.topics) * 100

        print(f"\nBERTopic results:")
        print(f"  Topics discovered: {n_topics}")
        print(f"  Outliers: {n_outliers} ({outlier_pct:.1f}%)")
        print(f"  Top topics:")
        for _, row in self.topic_info.head(6).iterrows():
            print(f"    Topic {row['Topic']}: {row['Name']} "
                  f"({row['Count']} docs)")

        return np.array(self.topics)

    def get_topic_keywords(self, top_n=10):
        """Get keywords for each topic. Returns dict of topic_id -> [(word, score)]."""
        if self.topic_model is None:
            raise RuntimeError("Model not fitted. Call fit_predict() first.")

        topics = self.topic_model.get_topics()
        result = {}
        for topic_id, words in topics.items():
            result[topic_id] = words[:top_n]
        return result

    def get_topic_labels(self):
        """Get auto-generated topic labels from BERTopic."""
        if self.topic_info is None:
            return {}
        return dict(zip(self.topic_info["Topic"], self.topic_info["Name"]))

    def save_model(self, output_path):
        """Save the BERTopic model."""
        if self.topic_model is None:
            raise RuntimeError("No model to save.")
        self.topic_model.save(output_path)
        print(f"BERTopic model saved to {output_path}")


class ClusterComparison:
    """Compare clustering quality metrics between KMeans and BERTopic."""

    @staticmethod
    def compute_metrics(embeddings, labels, method_name="unknown"):
        """Compute silhouette, calinski-harabasz, and davies-bouldin."""
        # Skip outliers (label == -1) for metrics
        valid_mask = labels >= 0
        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]

        n_clusters = len(set(valid_labels))
        n_outliers = int(np.sum(~valid_mask))

        if n_clusters < 2 or len(valid_embeddings) < n_clusters:
            logger.warning(
                "%s: not enough valid clusters for metrics (got %d)",
                method_name, n_clusters
            )
            return {
                "method": method_name,
                "n_clusters": n_clusters,
                "n_outliers": n_outliers,
                "silhouette": None,
                "calinski_harabasz": None,
                "davies_bouldin": None,
            }

        # Subsample for silhouette (expensive on large N)
        sample_size = min(10000, len(valid_embeddings))
        if sample_size < len(valid_embeddings):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(valid_embeddings), size=sample_size, replace=False)
            sample_emb = valid_embeddings[idx]
            sample_labels = valid_labels[idx]
        else:
            sample_emb = valid_embeddings
            sample_labels = valid_labels

        sil = silhouette_score(sample_emb, sample_labels)
        ch = calinski_harabasz_score(valid_embeddings, valid_labels)
        db = davies_bouldin_score(valid_embeddings, valid_labels)

        result = {
            "method": method_name,
            "n_clusters": n_clusters,
            "n_outliers": n_outliers,
            "silhouette": round(float(sil), 4),
            "calinski_harabasz": round(float(ch), 2),
            "davies_bouldin": round(float(db), 4),
        }

        print(f"\n{method_name} cluster quality:")
        print(f"  Clusters: {n_clusters}, Outliers: {n_outliers}")
        print(f"  Silhouette Score: {sil:.4f}  (higher is better, range [-1, 1])")
        print(f"  Calinski-Harabasz: {ch:.2f}  (higher is better)")
        print(f"  Davies-Bouldin: {db:.4f}  (lower is better)")

        return result

    @staticmethod
    def compare(embeddings, kmeans_labels, bertopic_labels):
        """Run metrics on both methods. Returns dict with both results."""
        km_metrics = ClusterComparison.compute_metrics(
            embeddings, kmeans_labels, "KMeans"
        )
        bt_metrics = ClusterComparison.compute_metrics(
            embeddings, bertopic_labels, "BERTopic"
        )

        return {"kmeans": km_metrics, "bertopic": bt_metrics}

    @staticmethod
    def save_results(results, output_path):
        """Save comparison results as JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Cluster comparison saved to {output_path}")


def cluster_complaints(df, embeddings, n_clusters=6, text_column='clean_text'):
    """Cluster complaints and extract keywords. Returns (df, keywords_dict)."""
    clusterer = KMeansClusterer(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(embeddings)
    df['cluster'] = cluster_labels

    cluster_keywords = clusterer.extract_keywords(df, text_column)
    
    return df, cluster_keywords


if __name__ == "__main__":
    # Quick test
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "processed_complaints.csv")
    embeddings_path = os.path.join(base_dir, "data", "processed", "embeddings.npy")
    
    df = pd.read_csv(data_path)
    embeddings = np.load(embeddings_path)
    
    df, keywords = cluster_complaints(df, embeddings, n_clusters=6)
    
    print(f"\nDone. Keywords:")
    for cluster_id, kw_list in keywords.items():
        print(f"  Cluster {cluster_id}: {kw_list}")
