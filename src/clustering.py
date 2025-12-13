from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle


class ComplaintClusterer:
    """Cluster complaints using KMeans and extract keywords."""
    
    def __init__(self, n_clusters=6, random_state=42):
        """
        Initialize clusterer.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_labels = None
    
    def fit_predict(self, embeddings):
        """
        Fit KMeans and predict cluster labels.
        
        Args:
            embeddings: numpy array of embeddings
        
        Returns:
            cluster labels array
        """
        print(f"Clustering {len(embeddings)} complaints into {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.cluster_labels = self.kmeans.fit_predict(embeddings)
        
        # Print cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} complaints ({count/len(self.cluster_labels)*100:.1f}%)")
        
        return self.cluster_labels
    
    def extract_keywords(self, df, text_column='clean_text', top_n=10):
        """
        Extract top keywords for each cluster using TF-IDF.
        
        Args:
            df: DataFrame with text and cluster labels
            text_column: Column containing text
            top_n: Number of top keywords per cluster
        
        Returns:
            dict mapping cluster_id to list of keywords
        """
        print(f"\nExtracting top {top_n} keywords per cluster...")
        cluster_keywords = {}
        
        for cluster_id in range(self.n_clusters):
            # Get texts from this cluster
            cluster_texts = df[df['cluster'] == cluster_id][text_column].tolist()
            
            if len(cluster_texts) == 0:
                cluster_keywords[cluster_id] = []
                continue
            
            # Extract keywords using TF-IDF
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
            except:
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


def cluster_complaints(df, embeddings, n_clusters=6, text_column='clean_text'):
    """
    Cluster complaints and extract keywords.
    
    Args:
        df: DataFrame with complaints
        embeddings: numpy array of embeddings
        n_clusters: Number of clusters
        text_column: Column containing text
    
    Returns:
        DataFrame with cluster labels, dict of keywords
    """
    clusterer = ComplaintClusterer(n_clusters=n_clusters)
    
    # Fit and predict
    cluster_labels = clusterer.fit_predict(embeddings)
    df['cluster'] = cluster_labels
    
    # Extract keywords
    cluster_keywords = clusterer.extract_keywords(df, text_column)
    
    return df, cluster_keywords


if __name__ == "__main__":
    # Example usage
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "processed_complaints.csv")
    embeddings_path = os.path.join(base_dir, "data", "processed", "embeddings.npy")
    
    df = pd.read_csv(data_path)
    embeddings = np.load(embeddings_path)
    
    df, keywords = cluster_complaints(df, embeddings, n_clusters=6)
    
    # Save updated dataframe with clusters
    df.to_csv(data_path, index=False)
    
    print(f"\nClustering complete!")
    print(f"Keywords by cluster:")
    for cluster_id, kw_list in keywords.items():
        print(f"  Cluster {cluster_id}: {kw_list}")
