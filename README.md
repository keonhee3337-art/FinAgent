# FinAgent — Multi-Agent Financial Analysis System

**Live:** [keonhee-finagent.streamlit.app](https://keonhee-finagent.streamlit.app)

A production multi-agent financial analysis system built with LangGraph, RAG, Text2SQL, FastAPI, and Streamlit. Answers natural language questions about Korean public companies using a three-agent pipeline that combines structured database queries with document-grounded reasoning.

---

## Architecture

```
User query
    │
    ▼
[SQL Agent]          → GPT-4o generates SQL → executes against SQLite
    │                  (Samsung, SK Hynix, LG Electronics — 2020–2024 financials)
    ▼
[RAG Agent]          → queries custom VectorDB → retrieves top-3 relevant chunks
    │                  (OpenAI text-embedding-3-small + cosine similarity)
    ▼
[Report Agent]       → GPT-4o synthesizes SQL results + RAG findings
    │                  → structured markdown: Key Findings, Financial Data, Market Context
    ▼
FastAPI backend      → POST /analyze endpoint
    │
Streamlit frontend   → 3 tabs: Final Report | SQL Results | RAG Findings
```

**Orchestration:** LangGraph `StateGraph` — shared `AgentState` (TypedDict) flows through nodes with typed edges. Linear pipeline: sql_agent → rag_agent → report_agent → END.

---

## Key Technical Decisions

### Custom VectorDB instead of ChromaDB
ChromaDB is incompatible with Python 3.14 due to a Pydantic v1 runtime type inference issue. Rather than downgrade Python or pin a broken dependency, a custom vector database was built from primitives:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Embeddings are generated via OpenAI `text-embedding-3-small`, stored as JSON, and retrieved via cosine similarity at query time. No external VectorDB dependency.

### Text2SQL with schema injection
The SQL agent injects the full SQLite schema into the GPT-4o system prompt. This gives the model exact column names and types, preventing hallucinated field names and improving SQL accuracy on first attempt.

### UNIQUE constraint on financial data
`init_db.py` uses `UNIQUE(company, year)` with `INSERT OR IGNORE` to prevent duplicate records when the database initialization script is run multiple times.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph, TypedDict AgentState) |
| Structured retrieval | Text2SQL — GPT-4o + SQLite |
| Semantic retrieval | RAG — custom VectorDB (OpenAI `text-embedding-3-small` + NumPy cosine similarity) |
| Report generation | GPT-4o |
| Backend | FastAPI |
| Frontend | Streamlit |
| Database | SQLite (Samsung Electronics, SK Hynix, LG Electronics, 2020–2024) |

---

## Data

Korean public company financials sourced from public disclosures:

| Company | Years | Fields |
|---------|-------|--------|
| Samsung Electronics | 2020–2024 | revenue, operating_profit, net_income, total_assets, total_liabilities, equity |
| SK Hynix | 2020–2024 | same |
| LG Electronics | 2020–2024 | same |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add API key
echo "OPENAI_API_KEY=your-key" > .env

# Initialize database + build vector store
python setup/init_db.py
python setup/build_vector_store.py

# Run Streamlit app
streamlit run app.py

# Or run FastAPI backend
uvicorn api:app --reload
```

---

## Project Structure

```
FinAgent/
├── agent/
│   ├── graph.py          # LangGraph pipeline (StateGraph, 3 nodes)
│   ├── sql_agent.py      # Text2SQL — schema injection + GPT-4o + SQLite
│   ├── rag_agent.py      # RAG — custom VectorDB query + GPT-4o
│   ├── report_agent.py   # Synthesis — structured markdown report
│   └── vector_store.py   # Custom VectorDB (embeddings + cosine similarity)
├── setup/
│   ├── init_db.py        # SQLite schema + seed data
│   └── build_vector_store.py
├── app.py                # Streamlit frontend
├── api.py                # FastAPI backend
└── requirements.txt
```

---

## Built By

**Keonhee** — Business Administration, Sungkyunkwan University (SKKU), South Korea.
Agentic AI developer building production systems at the intersection of AI engineering and business strategy.

**Other projects:**
- [DART Financial App](https://keonhee-strategy.streamlit.app) — Samsung data via DART API → SQLite → RAG → GPT-4o
- [github.com/keonhee3337-art](https://github.com/keonhee3337-art)
