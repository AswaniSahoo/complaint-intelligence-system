# Customer Complaint Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

## 🚀 Overview

The **Customer Complaint Intelligence System** is a GenAI-powered application that analyzes, clusters, and extracts insights from customer complaints. By using advanced NLP techniques like embeddings and Large Language Models (LLMs), it transforms raw text into actionable intelligence.

**Live Demo:** [Click Here to View App](https://complaint-intelligence-system.streamlit.app/)

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

