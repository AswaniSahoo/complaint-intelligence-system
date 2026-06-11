import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
import sys
from pathlib import Path

# Resolve project root (parent of app/) and add to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import ComplaintEmbedder, EMBEDDING_MODELS
from src.rag import ComplaintRAG, ComplaintQA


# Page config
st.set_page_config(
    page_title="Complaint Intelligence System",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E0E0E0;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #8B8FA3;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #C0C4D0;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .glass-card {
        background: rgba(26, 31, 43, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4A9EFF;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #8B8FA3;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] {
        background: rgba(26, 31, 43, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    div[data-testid="stMetric"] label {
        color: #8B8FA3 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #E0E0E0 !important;
        font-weight: 600;
    }
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        background: rgba(26, 31, 43, 0.5) !important;
    }
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
        margin: 1.5rem 0 !important;
    }
    section[data-testid="stSidebar"] {
        background: #0E1117;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)


# -- Data Loading Helpers ------------------------------------------------------

@st.cache_data
def load_data():
    """Load processed complaints data."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "processed_complaints.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


@st.cache_resource
def load_embeddings():
    """Load embeddings, preferring MiniLM for the RAG system."""
    # Try MiniLM first (200K run output), fall back to legacy name
    for name in ["embeddings_minilm.npy", "embeddings.npy"]:
        path = PROJECT_ROOT / "data" / "processed" / name
        if path.exists():
            return np.load(path)
    raise FileNotFoundError("No embedding file found in data/processed/")


@st.cache_resource
def initialize_rag(_embeddings):
    """Initialize RAG system."""
    dim = _embeddings.shape[1]
    rag = ComplaintRAG(_embeddings, dim)
    return rag


@st.cache_resource
def initialize_embedder():
    """Initialize embedder for query encoding."""
    return ComplaintEmbedder()


@st.cache_data
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
    st.markdown('<div class="main-header">Complaint Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">200K CFPB complaints analyzed with dual embeddings, multi-method clustering, and benchmarked retrieval</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Complaints", f"{len(df):,}")
    with col2:
        st.metric("Products", df['product'].nunique())
    with col3:
        if 'cluster' in df.columns:
            st.metric("KMeans Clusters", df['cluster'].nunique())
        else:
            st.metric("Clusters", "N/A")
    with col4:
        if 'topic' in df.columns:
            n_topics = df['topic'].nunique()
            st.metric("BERTopic Topics", n_topics)
        else:
            st.metric("Topics", "N/A")

    st.markdown("---")

    # Two columns layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Top Products by Complaint Volume</div>', unsafe_allow_html=True)
        product_counts = df['product'].value_counts().head(10)
        fig = px.bar(
            x=product_counts.values,
            y=product_counts.index,
            orientation='h',
            labels={'x': 'Complaints', 'y': ''},
            color=product_counts.values,
            color_continuous_scale=[[0, '#1a3a5c'], [1, '#4A9EFF']],
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Top Issues</div>', unsafe_allow_html=True)
        issue_counts = df['issue'].value_counts().head(10)
        fig = px.bar(
            x=issue_counts.values,
            y=issue_counts.index,
            orientation='h',
            labels={'x': 'Complaints', 'y': ''},
            color=issue_counts.values,
            color_continuous_scale=[[0, '#3a1a1a'], [1, '#FF6B6B']],
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Time series
    st.markdown('<div class="section-title">Complaint Volume Over Time</div>', unsafe_allow_html=True)
    df_time = df.dropna(subset=['date']).set_index('date').resample('ME').size().reset_index()
    df_time.columns = ['date', 'count']

    fig = px.area(
        df_time, x='date', y='count',
        labels={'date': '', 'count': 'Complaints'},
    )
    fig.update_traces(
        fill='tozeroy',
        fillcolor='rgba(74, 158, 255, 0.1)',
        line=dict(color='#4A9EFF', width=2),
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Cluster distribution
    if 'cluster' in df.columns:
        st.markdown('<div class="section-title">Cluster Distribution</div>', unsafe_allow_html=True)
        cluster_counts = df['cluster'].value_counts().sort_index()
        fig = px.bar(
            x=[f"Cluster {c}" for c in cluster_counts.index],
            y=cluster_counts.values,
            labels={'x': '', 'y': 'Complaints'},
            color=cluster_counts.values,
            color_continuous_scale=[[0, '#1a3a2a'], [1, '#00CC96']],
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def clusters_page(df):
    """Cluster drilldown page."""
    st.markdown('<div class="main-header">Cluster Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Drill into each cluster to understand recurring complaint patterns</div>', unsafe_allow_html=True)

    st.markdown("---")

    if 'cluster' not in df.columns:
        st.warning("Clustering has not been performed yet. Run the pipeline first.")
        return

    cluster_id = st.selectbox(
        "Select Cluster",
        options=sorted(df['cluster'].unique()),
        format_func=lambda x: f"Cluster {x}"
    )

    cluster_df = df[df['cluster'] == cluster_id]

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Complaints", f"{len(cluster_df):,}")
    with col2:
        pct = (len(cluster_df) / len(df)) * 100
        st.metric("Share of Total", f"{pct:.1f}%")
    with col3:
        st.metric("Unique Products", cluster_df['product'].nunique())

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Top Products</div>', unsafe_allow_html=True)
        product_counts = cluster_df['product'].value_counts().head(5)
        fig = px.bar(
            x=product_counts.values, y=product_counts.index,
            orientation='h', labels={'x': 'Count', 'y': ''},
            color=product_counts.values,
            color_continuous_scale=[[0, '#1a3a5c'], [1, '#4A9EFF']],
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=280, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Top Issues</div>', unsafe_allow_html=True)
        issue_counts = cluster_df['issue'].value_counts().head(5)
        fig = px.bar(
            x=issue_counts.values, y=issue_counts.index,
            orientation='h', labels={'x': 'Count', 'y': ''},
            color=issue_counts.values,
            color_continuous_scale=[[0, '#3a2a1a'], [1, '#FFB347']],
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=280, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Sample complaints
    st.markdown('<div class="section-title">Sample Complaints</div>', unsafe_allow_html=True)

    sample_size = min(5, len(cluster_df))
    sample_df = cluster_df.sample(n=sample_size, random_state=42)

    for _, row in sample_df.iterrows():
        with st.expander(f"{row['product']} | {str(row['issue'])[:60]}"):
            st.write(f"**Product:** {row['product']}")
            st.write(f"**Issue:** {row['issue']}")
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'
            st.write(f"**Date:** {date_str}")
            text = str(row['complaint_text'])
            st.text(text[:500] + "..." if len(text) > 500 else text)


def viewer_page(df):
    """Complaint viewer with filters."""
    st.markdown('<div class="main-header">Complaint Viewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse and filter individual complaints</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        products = ['All'] + sorted(df['product'].unique().tolist())
        selected_product = st.selectbox("Filter by Product", products)
    with col2:
        if 'cluster' in df.columns:
            clusters = ['All'] + [f"Cluster {c}" for c in sorted(df['cluster'].unique())]
            selected_cluster = st.selectbox("Filter by Cluster", clusters)
        else:
            selected_cluster = 'All'
    with col3:
        search_text = st.text_input("Search in text", placeholder="e.g. credit card")

    # Apply filters
    filtered_df = df.copy()

    if selected_product != 'All':
        filtered_df = filtered_df[filtered_df['product'] == selected_product]

    if selected_cluster != 'All' and 'cluster' in df.columns:
        cluster_num = int(selected_cluster.split()[-1])
        filtered_df = filtered_df[filtered_df['cluster'] == cluster_num]

    if search_text:
        mask = filtered_df['clean_text'].str.contains(search_text.lower(), na=False)
        filtered_df = filtered_df[mask]

    st.write(f"Showing {len(filtered_df):,} complaints")

    # Display table
    display_columns = ['date', 'product', 'issue']
    if 'cluster' in df.columns:
        display_columns.append('cluster')

    display_df = filtered_df[display_columns].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df.head(200), use_container_width=True, height=400)

    # Detail view
    st.markdown("---")
    st.markdown('<div class="section-title">View Full Complaint</div>', unsafe_allow_html=True)

    if len(filtered_df) > 0:
        selected_idx = st.number_input(
            "Row number",
            min_value=0, max_value=min(len(filtered_df)-1, 199), value=0
        )
        row = filtered_df.iloc[selected_idx]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.text_area("", value=str(row['complaint_text']), height=250, disabled=True)
        with col2:
            st.write(f"**Product:** {row['product']}")
            st.write(f"**Issue:** {row['issue']}")
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'
            st.write(f"**Date:** {date_str}")
            if 'cluster' in row.index:
                st.write(f"**Cluster:** {row['cluster']}")
            if 'topic' in row.index:
                st.write(f"**Topic:** {row['topic']}")


def qa_page(df, rag, embedder):
    """Search page for natural language queries."""
    st.markdown('<div class="main-header">Semantic Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search complaints using natural language | powered by FAISS vector similarity</div>', unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("Example queries"):
        st.markdown("""
        - What are the main credit card billing issues?
        - Show complaints about identity theft
        - Find issues with mortgage loan modification
        - Debt collector harassment complaints
        - Bank account overdraft fee problems
        """)

    query = st.text_input("Enter your question:", placeholder="e.g., unauthorized transactions on my account")

    col1, col2 = st.columns([1, 5])
    with col1:
        k_results = st.slider("Results", min_value=3, max_value=20, value=5)

    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                qa = ComplaintQA(rag, embedder, df)
                response = qa.answer_query(query, k=k_results)

                st.success(f"Found {response['count']} relevant complaints")

                for i, result in enumerate(response['results'], 1):
                    sim_pct = result['similarity'] * 100

                    with st.expander(f"#{i} | {result['product']} ({sim_pct:.1f}% match)"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            text = result['complaint_text']
                            st.text(text[:500] + "..." if len(text) > 500 else text)
                        with col2:
                            st.write(f"**Product:** {result['product']}")
                            st.write(f"**Issue:** {result['issue']}")
                            st.write(f"**Date:** {result['date']}")
                            st.write(f"**Similarity:** {sim_pct:.1f}%")

                # Insights
                st.markdown("---")
                st.markdown('<div class="section-title">Quick Insights</div>', unsafe_allow_html=True)

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


# -- Comparison Pages ----------------------------------------------------------

def embedding_comparison_page(df):
    """Embedding model comparison."""
    st.markdown('<div class="main-header">Embedding Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">MiniLM (384d, 2022 baseline) vs BGE (768d, 2024 SOTA) | benchmarked on 5K sample</div>', unsafe_allow_html=True)

    st.markdown("---")

    benchmark = load_json_results("embedding_benchmark.json")
    if benchmark is None:
        st.warning("No embedding benchmark results found. Run: `python run_pipeline.py --model both`")
        return

    # Model overview cards
    st.markdown('<div class="section-title">Model Overview</div>', unsafe_allow_html=True)
    model_keys = [k for k in benchmark if k != "cross_model"]

    cols = st.columns(len(model_keys))
    for col, key in zip(cols, model_keys):
        m = benchmark[key]
        with col:
            st.markdown(f"**{m['model_name']}**")
            st.metric("Dimension", m['embedding_dim'])
            st.metric("Throughput", f"{m['throughput_texts_per_sec']:.0f} texts/sec")
            st.metric("Memory", f"{m['memory_mb']:.1f} MB")

    st.markdown("---")

    # Cosine similarity
    st.markdown('<div class="section-title">Cosine Similarity Distribution</div>', unsafe_allow_html=True)
    sim_data = []
    for key in model_keys:
        sim = benchmark[key]["cosine_similarity"]
        sim_data.append({
            "Model": benchmark[key]["model_name"],
            "Mean": f"{sim['mean']:.4f}",
            "Std": f"{sim['std']:.4f}",
            "P25": f"{sim['p25']:.4f}",
            "P50": f"{sim['p50']:.4f}",
            "P75": f"{sim['p75']:.4f}",
        })
    st.dataframe(pd.DataFrame(sim_data), use_container_width=True, hide_index=True)

    st.info(
        "Lower mean cosine similarity = better differentiation. "
        "A model where all embeddings cluster near 1.0 is not distinguishing content."
    )

    # Cluster separation
    has_cluster = any(benchmark[k].get("cluster_separation") for k in model_keys)
    if has_cluster:
        st.markdown('<div class="section-title">Cluster Separation</div>', unsafe_allow_html=True)
        sep_data = []
        for key in model_keys:
            cs = benchmark[key].get("cluster_separation", {})
            if cs:
                sep_data.append({
                    "Model": benchmark[key]["model_name"],
                    "Intra-cluster Sim": f"{cs.get('intra_cluster_sim', 0):.4f}",
                    "Inter-cluster Sim": f"{cs.get('inter_cluster_sim', 0):.4f}",
                    "Separation Gap": f"{cs.get('separation', 0):.4f}",
                })
        if sep_data:
            st.dataframe(pd.DataFrame(sep_data), use_container_width=True, hide_index=True)
            st.info(
                "Separation Gap = Intra - Inter cluster similarity. "
                "Higher = embeddings within same cluster are more similar than across clusters."
            )

    # Cross-model overlap
    if "cross_model" in benchmark:
        st.markdown("---")
        st.markdown('<div class="section-title">Cross-Model Agreement</div>', unsafe_allow_html=True)
        cm = benchmark["cross_model"]
        st.metric(
            "Top-10 Neighbor Overlap",
            f"{cm['top10_overlap_mean']:.1%}",
            help="How often both models agree on the 10 most similar texts for a query"
        )
        st.write(cm.get("interpretation", ""))


def clustering_comparison_page(df):
    """KMeans vs BERTopic comparison."""
    st.markdown('<div class="main-header">Clustering Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">KMeans (fixed k=6, centroid-based) vs BERTopic (HDBSCAN, automatic topic discovery)</div>', unsafe_allow_html=True)

    st.markdown("---")

    comparison = load_json_results("cluster_comparison.json")
    if comparison is None:
        st.warning("No clustering comparison results found. Run: `python run_pipeline.py --clustering both`")
        return

    # Side-by-side metrics
    st.markdown('<div class="section-title">Quality Metrics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for col, method_key in [(col1, "kmeans"), (col2, "bertopic")]:
        m = comparison.get(method_key, {})
        with col:
            st.markdown(f"**{m.get('method', method_key)}**")
            st.metric("Clusters Found", m.get("n_clusters", "N/A"))
            st.metric("Outliers", f"{m.get('n_outliers', 0):,}")

            sil = m.get("silhouette")
            if sil is not None:
                st.metric("Silhouette Score", f"{sil:.4f}",
                          help="Range [-1, 1]. Higher = better separation.")
            db = m.get("davies_bouldin")
            if db is not None:
                st.metric("Davies-Bouldin", f"{db:.4f}",
                          help="Lower = less cluster overlap.")

    # Visual comparison
    st.markdown("---")
    st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)

    from src.visualizer import EmbeddingVisualizer
    viz = EmbeddingVisualizer()
    fig = viz.plot_cluster_comparison(comparison)
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    st.markdown("---")
    st.markdown('<div class="section-title">Method Comparison</div>', unsafe_allow_html=True)

    km = comparison.get("kmeans", {})
    bt = comparison.get("bertopic", {})

    comparison_df = pd.DataFrame({
        "Aspect": ["Method", "Clusters", "Outliers", "Silhouette", "Davies-Bouldin"],
        "KMeans": [
            f"Centroid-based, fixed k={km.get('n_clusters', '?')}",
            str(km.get("n_clusters", "?")),
            "0 (forces all points)",
            str(km.get("silhouette", "N/A")),
            str(km.get("davies_bouldin", "N/A")),
        ],
        "BERTopic": [
            "HDBSCAN density-based, auto-k",
            str(bt.get("n_clusters", "?")),
            f"{bt.get('n_outliers', '?'):,} ({bt.get('n_outliers', 0) / len(df) * 100:.1f}%)",
            str(bt.get("silhouette", "N/A")),
            str(bt.get("davies_bouldin", "N/A")),
        ],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # BERTopic topics
    if 'topic' in df.columns:
        st.markdown("---")
        st.markdown('<div class="section-title">BERTopic Discovered Topics</div>', unsafe_allow_html=True)
        topic_counts = df['topic'].value_counts().head(15)
        fig = px.bar(
            x=topic_counts.values,
            y=[f"Topic {t}" for t in topic_counts.index],
            orientation='h',
            labels={'x': 'Documents', 'y': ''},
            color=topic_counts.values,
            color_continuous_scale=[[0, '#1a3a2a'], [1, '#00CC96']],
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def retrieval_benchmark_page():
    """Retrieval latency benchmark."""
    st.markdown('<div class="main-header">Retrieval Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Latency comparison across 4 retrieval strategies on 200K corpus, 20 queries</div>', unsafe_allow_html=True)

    st.markdown("---")

    results = load_json_results("retrieval_benchmark.json")
    if results is None:
        st.warning("No retrieval benchmark results found. Run: `python run_pipeline.py --benchmark`")
        return

    # Summary table
    st.markdown('<div class="section-title">Latency Summary</div>', unsafe_allow_html=True)

    table_data = []
    for name, metrics in results.items():
        if "error" in metrics or "latency" not in metrics:
            continue
        lat = metrics["latency"]
        table_data.append({
            "Retriever": name,
            "p50 (ms)": f"{lat['p50_ms']:.1f}",
            "p95 (ms)": f"{lat['p95_ms']:.1f}",
            "p99 (ms)": f"{lat['p99_ms']:.1f}",
            "Mean (ms)": f"{lat['mean_ms']:.1f}",
        })

    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Bar chart
    st.markdown("---")
    st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)

    from src.visualizer import EmbeddingVisualizer
    viz = EmbeddingVisualizer()
    fig = viz.plot_retrieval_comparison(results)
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("---")
    st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)

    st.markdown("""
    - **Vector (FAISS)**: Pure nearest-neighbor search, no text processing overhead
    - **BM25**: Term-frequency matching, scales linearly with corpus size
    - **Hybrid (RRF)**: Runs both Vector + BM25 and fuses results via Reciprocal Rank Fusion
    - **Reranked Hybrid**: Adds a cross-encoder pass (ms-marco-MiniLM-L-6-v2) that re-scores candidates with full cross-attention

    The production pattern: use Hybrid for broad candidate generation, then Reranked for precision-critical queries.
    The cross-encoder step typically costs 200-400ms additional latency for higher retrieval quality.
    """)


# -- Main App ------------------------------------------------------------------

def main():
    """Main application."""

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Clusters",
            "Complaint Viewer",
            "Semantic Search",
            "Embedding Comparison",
            "Clustering Comparison",
            "Retrieval Benchmark",
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Stack")
    st.sidebar.markdown("""
    - **Embeddings:** MiniLM / BGE
    - **Clustering:** KMeans / BERTopic
    - **Retrieval:** FAISS + BM25 + Reranking
    - **Viz:** Plotly + UMAP
    """)

    st.sidebar.markdown("---")
    st.sidebar.caption("200K CFPB complaints | T4 GPU")

    # Load data
    try:
        df = load_data()

        if page == "Overview":
            overview_page(df)

        elif page == "Clusters":
            clusters_page(df)

        elif page == "Complaint Viewer":
            viewer_page(df)

        elif page == "Semantic Search":
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

    except FileNotFoundError:
        st.error("Data files not found. Run the pipeline first:")
        st.code("python run_pipeline.py --model both --clustering both --benchmark")


if __name__ == "__main__":
    main()
