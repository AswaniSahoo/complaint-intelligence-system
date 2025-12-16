# Customer Complaint Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

## 🚀 Overview

The **Customer Complaint Intelligence System** is a GenAI-powered application that analyzes, clusters, and extracts insights from customer complaints. By using advanced NLP techniques like embeddings and Large Language Models (LLMs), it transforms raw text into actionable intelligence.

**Live Demo:** [Link to your deployed app here]

## ✨ Key Features

- **Automated Cleaning**: Preprocesses raw complaint text automatically.
- **Semantic Clustering**: Groups similar complaints using sentence embeddings and KMeans.
- **AI Summarization**: Generates concise summaries and categories using Gemini or Groq.
- **Smart Search**: Find specific complaints using natural language queries (RAG).
- **Interactive Dashboard**: Explore data through a user-friendly Streamlit interface.

## 🛠️ Tech Stack

- **Python** (Logic & Data Processing)
- **Streamlit** (User Interface)
- **Sentence Transformers** (Embeddings)
- **FAISS** (Vector Database)
- **Gemini API / Groq API** (LLM Intelligence)

## 📸 Screenshots

### Dashboard
![Dashboard](screenshots/overview.png)

### Clusters
![Clusters](screenshots/clusters.png)

### Smart Search
![Search](screenshots/aisearch.png)

## ⚡ Getting Started

### 1. Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/customer-complaint-intelligence.git
cd customer-complaint-intelligence
pip install -r requirements.txt
```

### 2. Setup Keys

Create a `.env` file in the root directory and add your API key:

```bash
GEMINI_API_KEY="your-gemini-api-key"
```

### 3. Run the App

Launch the dashboard locally:

```bash
streamlit run app/app.py
```

Open your browser to `http://localhost:8501`.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

<<<<<<< HEAD
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
=======
Built as a portfolio project to demonstrate End-to-End GenAI development.
>>>>>>> d926920 (Optimize project: Clean docs, add community assets & screenshots)
