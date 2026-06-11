import logging

import faiss
import numpy as np
import pandas as pd
import pickle
import os

from src.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class ComplaintRAG:
    """Retrieval-Augmented Generation system for complaint search."""
    
    def __init__(self, embeddings=None, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.df = None
        
        if embeddings is not None:
            self.build_index(embeddings)
    
    def build_index(self, embeddings):
        """Build FAISS index from embeddings."""
        print(f"Building FAISS index with {len(embeddings)} vectors...")
        
        # Normalize for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query_embedding, k=5):
        """Search for k most similar complaints. Returns (distances, indices)."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
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
        self.rag = rag_system
        self.embedder = embedder
        self.df = df
        self.llm = llm_summarizer
    
    def answer_query(self, query, k=5, return_full_context=False):
        """Search complaints by query. Returns dict with results."""
        print(f"Searching for: '{query}'")
        
        query_embedding = self.embedder.encode([query], show_progress=False)[0]
        
        distances, indices = self.rag.search(query_embedding, k=k)
        
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
            

            if 'llm_summary' in complaint.index and pd.notna(complaint.get('llm_summary')):
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
        """Get aggregate patterns from top-k results."""
        response = self.answer_query(query, k=k)
        results = response['results']
        

        products = [r['product'] for r in results]
        issues = [r['issue'] for r in results]
        
        insights = {
            'query': query,
            'total_found': len(results),
            'top_products': pd.Series(products).value_counts().head(3).to_dict(),
            'top_issues': pd.Series(issues).value_counts().head(3).to_dict(),
            'sample_complaints': [r['complaint_text'][:200] for r in results[:3]]
        }
        
        if results and 'category' in results[0]:
            categories = [r['category'] for r in results]
            urgencies = [r['urgency'] for r in results]
            insights['top_categories'] = pd.Series(categories).value_counts().to_dict()
            insights['urgency_breakdown'] = pd.Series(urgencies).value_counts().to_dict()
        
        return insights


def build_rag_system(embeddings, df, embedding_dim=384):
    """Build a ComplaintRAG instance with FAISS index."""
    rag = ComplaintRAG(embeddings, embedding_dim)
    return rag


class RetrieverQA:
    """Unified QA system that works with any BaseRetriever."""

    def __init__(self, retriever: BaseRetriever, df: pd.DataFrame):
        self.retriever = retriever
        self.df = df

    def answer_query(self, query: str, k: int = 5):
        """Query any retriever and return standardized results."""
        results, latency_ms = self.retriever.retrieve_with_timing(query, k=k)

        response = {
            "query": query,
            "retriever": self.retriever.name,
            "results": [r.to_dict() for r in results],
            "count": len(results),
            "latency_ms": round(latency_ms, 2),
        }
        return response

    def get_insights(self, query: str, k: int = 10):
        """Get aggregate insights from retrieval results."""
        response = self.answer_query(query, k=k)
        results = response["results"]

        if not results:
            return {"query": query, "total_found": 0}

        products = [r["product"] for r in results]
        issues = [r["issue"] for r in results]

        insights = {
            "query": query,
            "retriever": self.retriever.name,
            "total_found": len(results),
            "latency_ms": response["latency_ms"],
            "top_products": pd.Series(products).value_counts().head(3).to_dict(),
            "top_issues": pd.Series(issues).value_counts().head(3).to_dict(),
            "sample_complaints": [r["text"][:200] for r in results[:3]],
        }
        return insights


if __name__ == "__main__":
    from pathlib import Path
    from src.embeddings import ComplaintEmbedder

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_path = PROJECT_ROOT / "data" / "processed" / "processed_complaints.csv"
    emb_path = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"

    df = pd.read_csv(data_path)
    embeddings = np.load(emb_path)

    # Quick test with legacy API
    rag = build_rag_system(embeddings, df)
    embedder = ComplaintEmbedder()
    qa = ComplaintQA(rag, embedder, df)

    response = qa.answer_query("credit card billing issues", k=5)
    print(f"\nFound {response['count']} relevant complaints")
    for i, result in enumerate(response['results'], 1):
        print(f"\n{i}. Product: {result['product']}")
        print(f"   Issue: {result['issue']}")
        print(f"   Similarity: {result['similarity']:.3f}")
