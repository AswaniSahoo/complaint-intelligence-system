import faiss
import numpy as np
import pandas as pd
import pickle
import os


class ComplaintRAG:
    """Retrieval-Augmented Generation system for complaint search."""
    
    def __init__(self, embeddings=None, embedding_dim=384):
        """
        Initialize RAG system.
        
        Args:
            embeddings: numpy array of complaint embeddings
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.df = None
        
        if embeddings is not None:
            self.build_index(embeddings)
    
    def build_index(self, embeddings):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of embeddings
        """
        print(f"Building FAISS index with {len(embeddings)} vectors...")
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query_embedding, k=5):
        """
        Search for similar complaints.
        
        Args:
            query_embedding: Embedding vector for the query
            k: Number of results to return
        
        Returns:
            distances, indices of top-k similar complaints
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Normalize query
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        return distances[0], indices[0]
    
    def save_index(self, output_path):
        """Save FAISS index to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        faiss.write_index(self.index, output_path)
        print(f"FAISS index saved to {output_path}")
    
    def load_index(self, input_path):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(input_path)
        print(f"FAISS index loaded from {input_path}")


class ComplaintQA:
    """Question-answering system over complaints."""
    
    def __init__(self, rag_system, embedder, df, llm_summarizer=None):
        """
        Initialize QA system.
        
        Args:
            rag_system: ComplaintRAG instance
            embedder: ComplaintEmbedder instance
            df: DataFrame with complaints
            llm_summarizer: LLMSummarizer instance (optional)
        """
        self.rag = rag_system
        self.embedder = embedder
        self.df = df
        self.llm = llm_summarizer
    
    def answer_query(self, query, k=5, return_full_context=False):
        """
        Answer a natural language query about complaints.
        
        Args:
            query: Natural language question
            k: Number of complaints to retrieve
            return_full_context: Return full complaint texts
        
        Returns:
            dict with query, retrieved complaints, and optional summary
        """
        print(f"Searching for: '{query}'")
        
        # Encode query
        query_embedding = self.embedder.encode([query], show_progress=False)[0]
        
        # Search
        distances, indices = self.rag.search(query_embedding, k=k)
        
        # Get results
        results = []
        for dist, idx in zip(distances, indices):
            complaint = self.df.iloc[idx]
            result = {
                'complaint_text': complaint['complaint_text'],
                'product': complaint.get('product', 'Unknown'),
                'issue': complaint.get('issue', 'Unknown'),
                'date': str(complaint.get('date', '')),
                'similarity': float(dist)
            }
            
            # Add LLM fields if available
            if 'llm_summary' in complaint:
                result['summary'] = complaint['llm_summary']
                result['category'] = complaint.get('llm_category', 'Other')
                result['urgency'] = complaint.get('llm_urgency', 'Medium')
            
            results.append(result)
        
        response = {
            'query': query,
            'results': results,
            'count': len(results)
        }
        
        return response
    
    def get_insights(self, query, k=10):
        """
        Get insights and patterns from retrieved complaints.
        
        Args:
            query: Natural language query
            k: Number of complaints to analyze
        
        Returns:
            dict with patterns and statistics
        """
        response = self.answer_query(query, k=k)
        results = response['results']
        
        # Extract patterns
        products = [r['product'] for r in results]
        issues = [r['issue'] for r in results]
        
        insights = {
            'query': query,
            'total_found': len(results),
            'top_products': pd.Series(products).value_counts().head(3).to_dict(),
            'top_issues': pd.Series(issues).value_counts().head(3).to_dict(),
            'sample_complaints': [r['complaint_text'][:200] for r in results[:3]]
        }
        
        if 'category' in results[0]:
            categories = [r['category'] for r in results]
            urgencies = [r['urgency'] for r in results]
            insights['top_categories'] = pd.Series(categories).value_counts().to_dict()
            insights['urgency_breakdown'] = pd.Series(urgencies).value_counts().to_dict()
        
        return insights


def build_rag_system(embeddings, df, embedding_dim=384):
    """
    Build complete RAG system.
    
    Args:
        embeddings: numpy array of embeddings
        df: DataFrame with complaints
        embedding_dim: Embedding dimension
    
    Returns:
        ComplaintRAG instance
    """
    rag = ComplaintRAG(embeddings, embedding_dim)
    return rag


if __name__ == "__main__":
    from embeddings import ComplaintEmbedder
    
    # Example usage
    df = pd.read_csv("../data/processed/processed_complaints.csv")
    embeddings = np.load("../data/processed/embeddings.npy")
    
    # Build RAG
    rag = build_rag_system(embeddings, df)
    
    # Create QA system
    embedder = ComplaintEmbedder()
    qa = ComplaintQA(rag, embedder, df)
    
    # Test query
    response = qa.answer_query("credit card billing issues", k=5)
    print(f"\nFound {response['count']} relevant complaints")
    for i, result in enumerate(response['results'], 1):
        print(f"\n{i}. Product: {result['product']}")
        print(f"   Issue: {result['issue']}")
        print(f"   Similarity: {result['similarity']:.3f}")
