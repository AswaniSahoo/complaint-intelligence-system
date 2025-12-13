# Customer Complaint Intelligence System

GenAI-powered system for analyzing, clustering, and extracting insights from customer complaints using embeddings, LLMs, and vector search.

## Features

- Automated complaint preprocessing and cleaning
- Semantic clustering using sentence embeddings
- AI-powered summarization and categorization (Gemini/Groq)
- Vector search with FAISS for complaint retrieval
- Natural language Q&A over complaints
- Interactive Streamlit dashboard

## Tech Stack

- Python 3.10+
- Sentence Transformers (all-MiniLM-L6-v2)
- KMeans clustering
- Gemini API / Groq API
- FAISS vector database
- Streamlit

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd GenAi
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
Create a `.env` file or set environment variables:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

## Usage

### 1. Process Data

Run the data exploration notebook or preprocessing script:

```bash
# Using notebook
jupyter notebook notebooks/analysis.ipynb

# Or using script
python src/preprocess.py
```

This will:
- Load the raw CFPB complaints dataset
- Filter rows with complaint narratives
- Sample 15,000 complaints
- Clean and normalize text
- Save to `data/processed/processed_complaints.csv`

### 2. Generate Embeddings

```bash
python src/embeddings.py
```

Creates sentence embeddings using `all-MiniLM-L6-v2` model.

### 3. Cluster Complaints

```bash
python src/clustering.py
```

Performs KMeans clustering (6 clusters) and extracts keywords.

### 4. Generate LLM Summaries (Optional)

```bash
python src/llm_utils.py
```

Uses Gemini or Groq API to:
- Generate 1-2 line summaries
- Assign categories (Billing/App Issue/Delivery/Support/Other)
- Determine urgency (Low/Medium/High)

### 5. Run Dashboard

```bash
streamlit run app/app.py
```

The dashboard will open at `http://localhost:8501`

## Dashboard Pages

### 1. Overview
- Total complaints and key metrics
- Top products and issues
- Time trends
- Category and urgency distribution

### 2. Clusters
- Drilldown into each cluster
- Cluster statistics and patterns
- Sample complaints from clusters

### 3. Complaint Viewer
- Browse all complaints
- Filter by product, category, urgency
- View full complaint details

### 4. Ask AI
- Natural language search over complaints
- Semantic similarity matching
- Quick insights and patterns

## Project Structure

```
GenAi/
├── data/
│   ├── raw/
│   │   └── complaints.csv
│   └── processed/
│       ├── processed_complaints.csv
│       └── embeddings.npy
├── src/
│   ├── preprocess.py        # Data cleaning
│   ├── embeddings.py         # Embedding generation
│   ├── clustering.py         # KMeans clustering
│   ├── llm_utils.py          # LLM summarization
│   └── rag.py                # FAISS + RAG
├── app/
│   └── app.py                # Streamlit dashboard
├── notebooks/
│   └── analysis.ipynb        # Data exploration
├── requirements.txt
└── README.md
```

## Dataset

This project uses the CFPB Consumer Complaint Database. The dataset contains:
- `Consumer complaint narrative`: The complaint text
- `Product`: Product category
- `Issue`: Specific issue type
- `Date received`: Complaint date

Sample size: 15,000 complaints with non-null narratives

## Model Details

### Embeddings
- Model: `all-MiniLM-L6-v2`
- Dimension: 384
- Library: sentence-transformers

### Clustering
- Algorithm: KMeans
- Clusters: 6
- Features: Sentence embeddings

### LLM
- Primary: Google Gemini 1.5 Flash
- Alternative: Groq (Llama 3.1 8B)
- Task: Summarization + Classification

### Vector Search
- Engine: FAISS
- Index: IndexFlatIP (cosine similarity)
- Retrieval: Top-k semantic search

## Evaluation

Simple evaluation metrics used:
- Cluster size distribution
- Manual quality check on sample complaints
- Response latency (target: <2s)
- User testing checklist

## Deployment to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Gemini API key

### Step-by-Step Deployment

1. **Prepare Your Repository**
   ```bash
   # Initialize git (if not already done)
   git init
   git add .
   git commit -m "Initial commit - Customer Complaint Intelligence System"
   ```

2. **Push to GitHub**
   ```bash
   # Create a new repository on GitHub, then:
   git remote add origin https://github.com/your-username/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set main file path: `app/app.py`
   - Click "Advanced settings"

4. **Add Secrets (IMPORTANT)**
   In the Secrets section, add:
   ```toml
   GEMINI_API_KEY = "your-actual-gemini-api-key-here"
   ```

5. **Deploy**
   - Click "Deploy!"
   - Wait 2-5 minutes for deployment
   - Your app will be live at: `https://your-app-name.streamlit.app`

### Important Notes
- The app will automatically download the embedding model on first run
- Processed data files (`processed_complaints.csv` and `embeddings.npy`) should be in your repository
- If files are too large (>100MB), use Git LFS or regenerate them on first run
- Environment variables are securely stored in Streamlit Cloud secrets

### Troubleshooting
- **App crashes on startup**: Check that all data files are present
- **Import errors**: Verify `requirements.txt` is complete
- **API errors**: Confirm `GEMINI_API_KEY` is set in secrets
- **Memory issues**: Streamlit Cloud free tier has 1GB RAM limit

## API Keys

The system supports two LLM providers:

**Gemini API** (Recommended):
- Sign up at: https://ai.google.dev/
- Free tier available in googleai studio

**Groq API**:
- Sign up at: https://console.groq.com/
- Fast inference with Llama models

## License

MIT License

## Author

Built as a GenAI + Data Science portfolio project demonstrating:
- End-to-end ML pipeline
- LLM integration
- Vector search/RAG
- Dashboard development
- Real-world data handling
