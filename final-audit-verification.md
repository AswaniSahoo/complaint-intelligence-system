# Project Audit & Verification Plan (final-audit-verification)

This plan outlines a deep-level audit and verification of the Complaint Intelligence System. The goal is to review the entire codebase for security, code style, performance, test coverage, and presentation, identifying any improvements needed to make this project stand out to recruiters.

## Project Type
- **Type**: BACKEND / WEB (Python RAG Pipeline + Streamlit Dashboard)

## User Review Required
> [!IMPORTANT]
> - **Linter Status**: The standard linter check (`ruff`) failed because it's not in the virtual environment. We can install `ruff` in the `.venv` to run automated code formatting checks, or perform a manual style audit.
> - **Data Scaling**: The app is currently deployed with a 15K sample for Streamlit Cloud hosting limits. The Colab pipeline is configured for the full 200K run. We should verify both configurations remain fully operational.

## Success Criteria
- [ ] 100% of unit and integration tests passing (87/87 tests).
- [ ] Security audit passes with zero critical vulnerabilities.
- [ ] Codebase audited for clean code patterns, caching optimizations, and proper error handling.
- [ ] Portfolio presentation elements (README, CONTRIBUTING, notebooks) checked and verified.

## Tech Stack
- **Backend/NLP**: Python, Pandas, FAISS, SentenceTransformers, BM25, Cross-Encoder
- **Web UI**: Streamlit, Plotly
- **Testing**: Pytest

---

## Proposed Tasks & Agent Assignments

### Task 1: Security & Dependency Audit
- **Agent**: `security-auditor`
- **Focus**: Review `requirements.txt` for outdated/vulnerable packages, scan for hardcoded secrets/API keys, and run `.agents/skills/vulnerability-scanner/scripts/security_scan.py`.
- **INPUT**: `requirements.txt`, `.env.example`, source code
- **OUTPUT**: Security report and vulnerability analysis
- **VERIFY**: Run security scan script

### Task 2: Code Quality & Style Audit
- **Agent**: `backend-specialist`
- **Focus**: Audit imports, function sizes, type hints, and code comments in `src/` and `app/`. Identify areas where Streamlit caching (`@st.cache_resource`, `@st.cache_data`) can be optimized.
- **INPUT**: `app/app.py`, `src/` modules
- **OUTPUT**: Code quality findings and concrete refactoring suggestions
- **VERIFY**: Successful verification of local app launch

### Task 3: Retrieval & Performance Audit
- **Agent**: `performance-optimizer`
- **Focus**: Review FAISS index creation, BM25 search parameters, and reciprocal rank fusion (RRF) logic. Look for latency bottlenecks (e.g. batch size settings for embeddings, FAISS query batching).
- **INPUT**: `src/retrievers/`, `src/evaluation/retrieval_benchmark.py`
- **OUTPUT**: Performance insights and latency optimizations
- **VERIFY**: Review `retrieval_benchmark.json` and latency charts in Streamlit

### Task 4: Test Suite & Edge Cases Audit
- **Agent**: `test-engineer`
- **Focus**: Verify test coverage of retrievers, edge cases (empty search query, invalid model selection, missing cache files), and preprocessing pipeline.
- **INPUT**: `tests/` directory
- **OUTPUT**: Execution of all 87 tests and edge case coverage analysis
- **VERIFY**: Run `pytest` and confirm 100% pass rate

### Task 5: Portfolio Presentation & Documentation Polish
- **Agent**: `documentation-writer`
- **Focus**: Review `README.md`, `CONTRIBUTING.md`, and the Colab notebooks to make sure everything is polished, easy to read, and professionally structured for recruiters.
- **INPUT**: `README.md`, `CONTRIBUTING.md`, `notebooks/`
- **OUTPUT**: Suggestions for documentation improvements
- **VERIFY**: Clickable links check and visual format check

---

## Phase X: Verification Checklist
- [x] Pytest suite: `python -m pytest` → ✅ 87/87 passed
- [x] Security Scan: `python .agents/skills/vulnerability-scanner/scripts/security_scan.py .` → ✅ Pass
- [ ] Streamlit Local Boot: `python -m streamlit run app/app.py` → ✅ Pass
- [ ] App Cloud Link: `https://complaint-intelligence-system.streamlit.app/` → ✅ Live
