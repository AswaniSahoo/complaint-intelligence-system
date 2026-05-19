import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path

# Resolve project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.embeddings import ComplaintEmbedder
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
</style>
""", unsafe_allow_html=True)


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


def main():
    """Main application."""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Clusters", "Complaint Viewer", "Ask AI"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This system uses GenAI to analyze customer complaints, "
        "identify patterns, and enable natural language search."
    )
    
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
    
    except FileNotFoundError as e:
        st.error("Data files not found. Please run the data processing pipeline first.")
        st.code("""
# Run these steps:
1. Process the data: python src/preprocess.py
2. Generate embeddings: python src/embeddings.py
3. Cluster complaints: python src/clustering.py
4. (Optional) Generate LLM summaries: python src/llm_utils.py
        """)


if __name__ == "__main__":
    main()
