import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

# Resolve project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.embeddings import ComplaintEmbedder, EMBEDDING_MODELS
from src.rag import ComplaintRAG, ComplaintQA
from src.llm_utils import LLMSummarizer


# Page config
st.set_page_config(
    page_title="Customer Complaint Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .comparison-highlight {
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# -- Data Loading Helpers ------------------------------------------------------

@st.cache_data
def load_data():
    """Load processed complaints data."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_complaints.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


@st.cache_resource
def load_embeddings():
    """Load embeddings."""
    return np.load(PROJECT_ROOT / "data" / "processed" / "embeddings.npy")


@st.cache_resource
def initialize_rag(_embeddings, embedding_dim=384):
    """Initialize RAG system."""
    rag = ComplaintRAG(_embeddings, embedding_dim)
    return rag


@st.cache_resource
def initialize_embedder():
    """Initialize embedder."""
    return ComplaintEmbedder()


def load_json_results(filename):
    """Load JSON results from data/results/ directory."""
    path = PROJECT_ROOT / "data" / "results" / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# -- Page Functions ------------------------------------------------------------

def overview_page(df):
    """Overview page with key metrics and trends."""
    st.markdown('<div class="main-header">Customer Complaint Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">GenAI-powered Summarization, Clustering & Insight Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Complaints", f"{len(df):,}")
    
    with col2:
        unique_products = df['product'].nunique()
        st.metric("Unique Products", unique_products)
    
    with col3:
        if 'cluster' in df.columns:
            st.metric("Identified Clusters", df['cluster'].nunique())
        else:
            st.metric("Identified Clusters", "N/A")
    
    with col4:
        if 'llm_urgency' in df.columns:
            high_urgency = (df['llm_urgency'] == 'High').sum()
            st.metric("High Urgency", high_urgency)
        else:
            st.metric("High Urgency", "N/A")
    
    st.markdown("---")
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Products by Complaint Volume")
        product_counts = df['product'].value_counts().head(10)
        fig = px.bar(
            x=product_counts.values,
            y=product_counts.index,
            orientation='h',
            labels={'x': 'Number of Complaints', 'y': 'Product'},
            color=product_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Issues")
        issue_counts = df['issue'].value_counts().head(10)
        fig = px.bar(
            x=issue_counts.values,
            y=issue_counts.index,
            orientation='h',
            labels={'x': 'Number of Complaints', 'y': 'Issue'},
            color=issue_counts.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.subheader("Complaint Trends Over Time")
    df_time = df.set_index('date').resample('ME').size().reset_index()
    df_time.columns = ['date', 'count']
    
    fig = px.line(
        df_time,
        x='date',
        y='count',
        labels={'date': 'Date', 'count': 'Number of Complaints'},
        markers=True
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    # Category distribution if available
    if 'llm_category' in df.columns:
        st.subheader("Complaint Categories (AI-Generated)")
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = df['llm_category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Category Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'llm_urgency' in df.columns:
                urgency_counts = df['llm_urgency'].value_counts()
                fig = px.pie(
                    values=urgency_counts.values,
                    names=urgency_counts.index,
                    title="Urgency Distribution",
                    color_discrete_sequence=['#90EE90', '#FFD700', '#FF6347']
                )
                st.plotly_chart(fig, use_container_width=True)


def clusters_page(df):
    """Clusters page with drilldown into each cluster."""
    st.markdown('<div class="main-header">Complaint Clusters</div>', unsafe_allow_html=True)
    st.markdown("Explore grouped complaints to identify recurring issues")
    
    st.markdown("---")
    
    if 'cluster' not in df.columns:
        st.warning("Clustering has not been performed yet. Please run the clustering pipeline first.")
        return
    
    # Cluster selection
    cluster_id = st.selectbox(
        "Select Cluster",
        options=sorted(df['cluster'].unique()),
        format_func=lambda x: f"Cluster {x}"
    )
    
    # Filter data for selected cluster
    cluster_df = df[df['cluster'] == cluster_id]
    
    # Cluster overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Complaints in Cluster", len(cluster_df))
    
    with col2:
        percentage = (len(cluster_df) / len(df)) * 100
        st.metric("Percentage of Total", f"{percentage:.1f}%")
    
    with col3:
        if 'llm_urgency' in cluster_df.columns:
            high_urgency = (cluster_df['llm_urgency'] == 'High').sum()
            st.metric("High Urgency", high_urgency)
    
    st.markdown("---")
    
    # Two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Products in Cluster")
        product_counts = cluster_df['product'].value_counts().head(5)
        fig = px.bar(
            x=product_counts.values,
            y=product_counts.index,
            orientation='h',
            color=product_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Issues in Cluster")
        issue_counts = cluster_df['issue'].value_counts().head(5)
        fig = px.bar(
            x=issue_counts.values,
            y=issue_counts.index,
            orientation='h',
            color=issue_counts.values,
            color_continuous_scale='Plasma'
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample complaints from cluster
    st.subheader("Sample Complaints from this Cluster")
    
    sample_size = min(5, len(cluster_df))
    sample_df = cluster_df.sample(n=sample_size)
    
    for idx, row in sample_df.iterrows():
        with st.expander(f"Complaint: {row['product']} - {row['issue'][:50]}..."):
            st.write(f"**Product:** {row['product']}")
            st.write(f"**Issue:** {row['issue']}")
            st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'}")
            
            if 'llm_summary' in row.index and pd.notna(row.get('llm_summary')):
                st.write(f"**AI Summary:** {row['llm_summary']}")
                st.write(f"**Category:** {row.get('llm_category', 'N/A')} | **Urgency:** {row.get('llm_urgency', 'N/A')}")
            
            st.write("**Full Text:**")
            st.text(row['complaint_text'][:500] + "..." if len(str(row['complaint_text'])) > 500 else row['complaint_text'])


def viewer_page(df):
    """Complaint viewer page with search and filters."""
    st.markdown('<div class="main-header">Complaint Viewer</div>', unsafe_allow_html=True)
    st.markdown("Browse and search through individual complaints")
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        products = ['All'] + sorted(df['product'].unique().tolist())
        selected_product = st.selectbox("Filter by Product", products)
    
    with col2:
        if 'llm_category' in df.columns:
            # Filter out NaN values before sorting
            categories = ['All'] + sorted(df['llm_category'].dropna().unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories)
        else:
            selected_category = 'All'
    
    with col3:
        if 'llm_urgency' in df.columns:
            urgencies = ['All'] + ['High', 'Medium', 'Low']
            selected_urgency = st.selectbox("Filter by Urgency", urgencies)
        else:
            selected_urgency = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_product != 'All':
        filtered_df = filtered_df[filtered_df['product'] == selected_product]
    
    if selected_category != 'All' and 'llm_category' in df.columns:
        filtered_df = filtered_df[filtered_df['llm_category'] == selected_category]
    
    if selected_urgency != 'All' and 'llm_urgency' in df.columns:
        filtered_df = filtered_df[filtered_df['llm_urgency'] == selected_urgency]
    
    st.write(f"Showing {len(filtered_df)} complaints")
    
    # Display table
    display_columns = ['date', 'product', 'issue']
    if 'llm_summary' in df.columns:
        display_columns.extend(['llm_summary', 'llm_category', 'llm_urgency'])
    
    display_df = filtered_df[display_columns].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    
    # Show table
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Detailed view
    st.markdown("---")
    st.subheader("View Full Complaint")
    
    if len(filtered_df) > 0:
        selected_idx = st.number_input(
            "Enter row number to view details",
            min_value=0,
            max_value=len(filtered_df)-1,
            value=0
        )
        
        row = filtered_df.iloc[selected_idx]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Full Complaint Text:**")
            st.text_area("", value=row['complaint_text'], height=300, disabled=True)
        
        with col2:
            st.write("**Details:**")
            st.write(f"**Product:** {row['product']}")
            st.write(f"**Issue:** {row['issue']}")
            st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'}")
            
            if 'llm_summary' in row.index and pd.notna(row.get('llm_summary')):
                st.write(f"**AI Summary:** {row['llm_summary']}")
                st.write(f"**Category:** {row.get('llm_category', 'N/A')}")
                st.write(f"**Urgency:** {row.get('llm_urgency', 'N/A')}")
            
            if 'cluster' in row.index:
                st.write(f"**Cluster:** {row['cluster']}")


def qa_page(df, rag, embedder):
    """Q&A page for natural language queries."""
    st.markdown('<div class="main-header">Ask AI</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about customer complaints in natural language")
    
    st.markdown("---")
    
    # Example queries
    with st.expander("Example Queries"):
        st.write("- What were the main issues in the last 30 days?")
        st.write("- Show me complaints about credit card billing")
        st.write("- Find issues related to account closure")
        st.write("- What are customers complaining about mortgage?")
        st.write("- Show me delivery or shipping problems")
    
    # Query input
    query = st.text_input("Enter your question:", placeholder="e.g., What are the main billing issues?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        k_results = st.slider("Number of results", min_value=3, max_value=20, value=5)
    
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                # Create QA system
                qa = ComplaintQA(rag, embedder, df)
                
                # Get results
                response = qa.answer_query(query, k=k_results)
                
                st.success(f"Found {response['count']} relevant complaints")
                
                # Display results
                for i, result in enumerate(response['results'], 1):
                    similarity_pct = result['similarity'] * 100
                    
                    with st.expander(f"Result {i} - {result['product']} (Similarity: {similarity_pct:.1f}%)"):
                        st.write(f"**Product:** {result['product']}")
                        st.write(f"**Issue:** {result['issue']}")
                        st.write(f"**Date:** {result['date']}")
                        
                        if 'summary' in result:
                            st.write(f"**AI Summary:** {result['summary']}")
                            st.write(f"**Category:** {result.get('category', 'N/A')} | **Urgency:** {result.get('urgency', 'N/A')}")
                        
                        st.write("**Complaint Text:**")
                        st.text(result['complaint_text'][:400] + "..." if len(result['complaint_text']) > 400 else result['complaint_text'])
                
                # Insights
                st.markdown("---")
                st.subheader("Quick Insights")
                
                insights = qa.get_insights(query, k=k_results)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Products:**")
                    for product, count in list(insights['top_products'].items())[:3]:
                        st.write(f"- {product}: {count}")
                
                with col2:
                    st.write("**Top Issues:**")
                    for issue, count in list(insights['top_issues'].items())[:3]:
                        st.write(f"- {issue}: {count}")
        else:
            st.warning("Please enter a question")


# -- NEW COMPARISON PAGES -----------------------------------------------------

def embedding_comparison_page(df):
    """Embedding model comparison page with UMAP plots and benchmarks."""
    st.markdown('<div class="main-header">Embedding Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("Side-by-side comparison of MiniLM (baseline) vs BGE (SOTA)")

    st.markdown("---")

    # Load benchmark results
    benchmark = load_json_results("embedding_benchmark.json")

    if benchmark is None:
        st.warning(
            "No embedding benchmark results found. "
            "Run: `python run_pipeline.py --model both`"
        )
        return

    # Summary metrics
    st.subheader("📊 Model Overview")
    model_keys = [k for k in benchmark if k != "cross_model"]

    cols = st.columns(len(model_keys))
    for col, key in zip(cols, model_keys):
        m = benchmark[key]
        with col:
            st.markdown(f"### {m['model_name']}")
            st.metric("Dimension", m['embedding_dim'])
            st.metric("Throughput", f"{m['throughput_texts_per_sec']:.0f} texts/sec")
            st.metric("Memory", f"{m['memory_mb']:.1f} MB")
            st.metric("Encoding Time", f"{m['encoding_time_sec']:.1f}s")

    st.markdown("---")

    # Cosine similarity comparison
    st.subheader("📈 Cosine Similarity Distribution")
    sim_data = []
    for key in model_keys:
        sim = benchmark[key]["cosine_similarity"]
        sim_data.append({
            "Model": benchmark[key]["model_name"],
            "Mean": sim["mean"],
            "Std": sim["std"],
            "P25": sim["p25"],
            "P50": sim["p50"],
            "P75": sim["p75"],
        })

    sim_df = pd.DataFrame(sim_data)
    st.dataframe(sim_df, use_container_width=True)

    st.info(
        "**Interpretation:** Lower mean cosine similarity indicates better "
        "spread in the embedding space — the model is better at distinguishing "
        "between different texts. A model where all embeddings cluster near "
        "1.0 similarity is not differentiating content well."
    )

    # Cluster separation (if available)
    has_cluster = any(
        benchmark[k].get("cluster_separation") for k in model_keys
    )
    if has_cluster:
        st.subheader("🎯 Cluster Separation")
        sep_data = []
        for key in model_keys:
            cs = benchmark[key].get("cluster_separation", {})
            if cs:
                sep_data.append({
                    "Model": benchmark[key]["model_name"],
                    "Intra-cluster Sim": cs.get("intra_cluster_sim", "N/A"),
                    "Inter-cluster Sim": cs.get("inter_cluster_sim", "N/A"),
                    "Separation Gap": cs.get("separation", "N/A"),
                })

        if sep_data:
            sep_df = pd.DataFrame(sep_data)
            st.dataframe(sep_df, use_container_width=True)

            st.info(
                "**Separation Gap** = Intra-cluster − Inter-cluster similarity. "
                "Higher is better — means embeddings within the same cluster are "
                "much more similar than embeddings across different clusters."
            )

    # Cross-model overlap
    if "cross_model" in benchmark:
        st.markdown("---")
        st.subheader("🔀 Cross-Model Agreement")
        cm = benchmark["cross_model"]
        st.metric(
            "Top-10 Neighbor Overlap",
            f"{cm['top10_overlap_mean']:.1%}",
            help="How often the two models agree on the 10 most similar texts for a given query"
        )
        st.write(cm.get("interpretation", ""))

    # UMAP visualization
    st.markdown("---")
    st.subheader("🗺️ Embedding Space (UMAP)")

    for key in ["minilm", "bge"]:
        umap_path = PROJECT_ROOT / "data" / "processed" / f"umap_{key}.npy"
        emb_path = PROJECT_ROOT / "data" / "processed" / f"embeddings_{key}.npy"

        if umap_path.exists():
            projection = np.load(umap_path)
            labels = df['cluster'].values[:len(projection)] if 'cluster' in df.columns else np.zeros(len(projection))

            fig = px.scatter(
                x=projection[:, 0], y=projection[:, 1],
                color=[str(l) for l in labels],
                title=f"{EMBEDDING_MODELS.get(key, {}).get('name', key)} — UMAP 2D",
                labels={"color": "Cluster"},
                opacity=0.5,
            )
            fig.update_traces(marker=dict(size=2))
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        elif emb_path.exists():
            st.info(
                f"UMAP projection for {key} not cached. "
                f"Run the pipeline notebook to generate it."
            )


def clustering_comparison_page(df):
    """Clustering comparison page: KMeans vs BERTopic."""
    st.markdown('<div class="main-header">Clustering Comparison</div>', unsafe_allow_html=True)
    st.markdown("KMeans (baseline, fixed-k) vs BERTopic (SOTA, automatic topic discovery)")

    st.markdown("---")

    comparison = load_json_results("cluster_comparison.json")

    if comparison is None:
        st.warning(
            "No clustering comparison results found. "
            "Run: `python run_pipeline.py --clustering both`"
        )
        return

    # Side-by-side metrics
    st.subheader("📊 Quality Metrics")

    col1, col2 = st.columns(2)

    for col, method_key, color in [
        (col1, "kmeans", "#636EFA"),
        (col2, "bertopic", "#00CC96"),
    ]:
        m = comparison.get(method_key, {})
        with col:
            st.markdown(f"### {m.get('method', method_key)}")
            st.metric("Clusters Found", m.get("n_clusters", "N/A"))
            st.metric("Outliers", m.get("n_outliers", 0))

            sil = m.get("silhouette")
            if sil is not None:
                st.metric("Silhouette Score", f"{sil:.4f}",
                          help="Range [-1, 1]. Higher = better cluster separation.")
            ch = m.get("calinski_harabasz")
            if ch is not None:
                st.metric("Calinski-Harabasz", f"{ch:.1f}",
                          help="Higher = denser, well-separated clusters.")
            db = m.get("davies_bouldin")
            if db is not None:
                st.metric("Davies-Bouldin", f"{db:.4f}",
                          help="Lower = better. Measures cluster overlap.")

    # Visual comparison chart
    st.markdown("---")
    st.subheader("📈 Visual Comparison")

    from src.visualizer import EmbeddingVisualizer
    viz = EmbeddingVisualizer()
    fig = viz.plot_cluster_comparison(comparison)
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("---")
    st.subheader("📝 Key Takeaways")

    km = comparison.get("kmeans", {})
    bt = comparison.get("bertopic", {})

    st.markdown(f"""
    | Aspect | KMeans | BERTopic |
    |--------|--------|----------|
    | **Method** | Centroid-based, fixed k={km.get('n_clusters', '?')} | Density-based (HDBSCAN), auto-discovers topics |
    | **Clusters** | {km.get('n_clusters', '?')} | {bt.get('n_clusters', '?')} |
    | **Outliers** | 0 (forces all points into clusters) | {bt.get('n_outliers', '?')} (noisy docs flagged) |
    | **Silhouette** | {km.get('silhouette', 'N/A')} | {bt.get('silhouette', 'N/A')} |
    """)

    # BERTopic topics (if data exists)
    if 'topic' in df.columns:
        st.markdown("---")
        st.subheader("🏷️ BERTopic Discovered Topics")
        topic_counts = df['topic'].value_counts().head(15)
        fig = px.bar(
            x=topic_counts.values,
            y=[f"Topic {t}" for t in topic_counts.index],
            orientation='h',
            labels={'x': 'Document Count', 'y': 'Topic'},
            color=topic_counts.values,
            color_continuous_scale='Viridis',
        )
        fig.update_layout(showlegend=False, height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


def retrieval_benchmark_page():
    """Retrieval benchmark comparison page."""
    st.markdown('<div class="main-header">Retrieval Benchmark</div>', unsafe_allow_html=True)
    st.markdown("Head-to-head latency comparison of all retrieval strategies")

    st.markdown("---")

    results = load_json_results("retrieval_benchmark.json")

    if results is None:
        st.warning(
            "No retrieval benchmark results found. "
            "Run: `python run_pipeline.py --benchmark`"
        )
        return

    # Summary table
    st.subheader("📊 Latency Summary")

    table_data = []
    for name, metrics in results.items():
        if "error" in metrics or "latency" not in metrics:
            continue
        lat = metrics["latency"]
        table_data.append({
            "Retriever": name,
            "p50 (ms)": lat["p50_ms"],
            "p95 (ms)": lat["p95_ms"],
            "p99 (ms)": lat["p99_ms"],
            "Mean (ms)": lat["mean_ms"],
            "Avg Results": metrics["avg_results_returned"],
        })

    if table_data:
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, use_container_width=True)

    # Bar chart comparison
    st.markdown("---")
    st.subheader("📈 Visual Comparison")

    from src.visualizer import EmbeddingVisualizer
    viz = EmbeddingVisualizer()
    fig = viz.plot_retrieval_comparison(results)
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("---")
    st.subheader("📝 What This Means")

    st.markdown("""
    **Key insights from the benchmark:**

    - **Vector (FAISS)**: Fastest — pure nearest-neighbor search with no text processing overhead
    - **BM25**: Very fast — no neural computation, pure term-frequency matching
    - **Hybrid (RRF)**: Slightly slower — runs both Vector + BM25 and fuses results
    - **Reranked Hybrid**: Slowest but highest quality — adds a cross-encoder pass that re-scores candidates with full cross-attention

    > **The production pattern**: Use Hybrid retrieval for broad candidate generation,
    > then Reranked for precision. The cross-encoder reranking step typically improves
    > retrieval quality by 18-42% at the cost of ~50-200ms additional latency.
    """)


# -- Main App ------------------------------------------------------------------

def main():
    """Main application."""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Clusters",
            "Complaint Viewer",
            "Ask AI",
            "─── Comparisons ───",
            "Embedding Comparison",
            "Clustering Comparison",
            "Retrieval Benchmark",
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This system uses GenAI to analyze customer complaints, "
        "identify patterns, and enable natural language search."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tech Stack")
    st.sidebar.markdown("""
    - **Embeddings:** MiniLM / BGE
    - **Clustering:** KMeans / BERTopic
    - **Retrieval:** FAISS + BM25 + Reranking
    - **LLM:** Gemini
    - **Viz:** Plotly + UMAP
    """)
    
    # Load data
    try:
        df = load_data()
        
        if page == "Overview":
            overview_page(df)
        
        elif page == "Clusters":
            clusters_page(df)
        
        elif page == "Complaint Viewer":
            viewer_page(df)
        
        elif page == "Ask AI":
            embeddings = load_embeddings()
            rag = initialize_rag(embeddings)
            embedder = initialize_embedder()
            qa_page(df, rag, embedder)

        elif page == "Embedding Comparison":
            embedding_comparison_page(df)

        elif page == "Clustering Comparison":
            clustering_comparison_page(df)

        elif page == "Retrieval Benchmark":
            retrieval_benchmark_page()

        elif page == "─── Comparisons ───":
            st.info("Select a specific comparison page from the sidebar.")
    
    except FileNotFoundError as e:
        st.error("Data files not found. Please run the data processing pipeline first.")
        st.code("""
# Run the full pipeline:
python run_pipeline.py --model both --clustering both --benchmark

# Or step by step:
python run_pipeline.py                    # Basic pipeline
python run_pipeline.py --model both       # Add BGE embeddings
python run_pipeline.py --clustering both  # Add BERTopic
python run_pipeline.py --benchmark        # Run retrieval benchmarks
        """)


if __name__ == "__main__":
    main()
