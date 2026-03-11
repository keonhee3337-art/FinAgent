# FinAgent — Multi-Agent Financial Analysis System

**Live demo:** [keonhee-finagent.streamlit.app](https://keonhee-finagent.streamlit.app)

Built by **Keonhee Kim**, Business Administration student at Sungkyunkwan University (SKKU), South Korea.

---

FinAgent is a production-deployed, multi-agent financial analysis system that automates structured financial data retrieval and document-grounded reasoning for Korean public companies. It answers natural language questions like "Compare Samsung Electronics and SK Hynix operating margins from 2020 to 2024" by routing queries through a three-agent LangGraph pipeline — without any human decision-making at each step.

**What it automates:** The workflow that would otherwise require an analyst to manually write SQL queries, search through financial documents, and synthesize findings into a structured report. FinAgent does this in one query, end-to-end.

---

## Architecture

```
User query (natural language)
    │
    ▼
[Router] — classifies query type
    │
    ├── [SQL Agent]     GPT-4o generates SQL → executes against SQLite
    │                   (Samsung Electronics, SK Hynix, LG Electronics — 2020–2024 financials)
    │
    ├── [RAG Agent]     Queries custom VectorDB → retrieves top-3 relevant document chunks
    │                   (OpenAI text-embedding-3-small + NumPy cosine similarity)
    │
    └── [Report Agent]  GPT-4o synthesizes SQL results + RAG findings
                        → Structured markdown: Key Findings, Financial Data, Market Context
    │
FastAPI backend   — POST /analyze endpoint
    │
Streamlit UI      — 3 tabs: Final Report | SQL Results | RAG Findings
```

**Orchestration:** LangGraph `StateGraph` with typed `AgentState` (TypedDict). Shared state flows through nodes with conditional edges. Streaming output via `graph.stream(stream_mode="values")`.

---

## Key Technical Decisions

### Custom VectorDB (not ChromaDB or Pinecone)

ChromaDB is incompatible with Python 3.14. Instead of downgrading or pinning a broken dependency, a custom vector database was built from NumPy primitives:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Embeddings (OpenAI `text-embedding-3-small`, 1536 dims) are stored as JSON and retrieved via cosine similarity at query time. Zero external VectorDB dependencies — runs anywhere Python runs.

### Text2SQL with schema injection

The SQL agent injects the full SQLite schema into the GPT-4o system prompt. This gives the model exact column names and types, eliminating hallucinated field names and achieving high SQL accuracy on the first attempt:

```python
system_prompt = f"""
You are a SQL expert. The database schema is:
{schema}
Generate a valid SQLite query for the following question.
"""
```

### LangGraph streaming for progressive rendering

Output streams as each agent completes, not after the full pipeline finishes:

```python
for state in graph.stream(initial_state, config=config, stream_mode="values"):
    if state.get("sql_result"):
        sql_placeholder.code(state["sql_result"])
    if state.get("rag_result"):
        rag_placeholder.markdown(state["rag_result"])
    if state.get("report"):
        report_placeholder.markdown(state["report"])
```

### UNIQUE constraint prevents duplicate data

`init_db.py` uses `UNIQUE(company, year)` with `INSERT OR IGNORE`, so the initialization script is idempotent — safe to re-run without duplicating financial records.

---

## Stack

| Layer | Technology |
|-------|------------|
| Agent orchestration | LangGraph `StateGraph` (v1.1+), typed AgentState |
| Structured retrieval | Text2SQL — GPT-4o + SQLite |
| Semantic retrieval | RAG — custom VectorDB (OpenAI embeddings + NumPy cosine similarity) |
| Report synthesis | GPT-4o |
| Backend | FastAPI (async, CORS-enabled) |
| Frontend | Streamlit (streaming, session state, cached graph) |
| Database | SQLite (Korean financial data, 2020–2024) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |

---

## Data

Korean public company financials sourced from public DART disclosures and annual reports:

| Company | Years | Fields |
|---------|-------|--------|
| Samsung Electronics (삼성전자) | 2020–2024 | revenue, operating_profit, net_income, total_assets, total_liabilities, equity |
| SK Hynix (SK하이닉스) | 2020–2024 | same |
| LG Electronics (LG전자) | 2020–2024 | same |

All figures in Korean Won (KRW, billions). Source: Korea Financial Supervisory Service DART system.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Initialize database and build vector store
python setup/init_db.py
python setup/build_vector_store.py

# 4a. Run Streamlit app (recommended)
streamlit run app.py

# 4b. Or run FastAPI backend separately
uvicorn api:app --reload --port 8000
```

**Requirements:** Python 3.10+, OpenAI API key.

---

## Project Structure

```
FinAgent/
├── agent/
│   ├── graph.py            # LangGraph pipeline — StateGraph, 3 nodes, streaming
│   ├── sql_agent.py        # Text2SQL — schema injection + GPT-4o + SQLite
│   ├── rag_agent.py        # RAG — custom VectorDB query + GPT-4o synthesis
│   ├── report_agent.py     # Report synthesis — structured markdown output
│   └── vector_store.py     # Custom VectorDB — embeddings + cosine similarity
├── setup/
│   ├── init_db.py          # SQLite schema creation + seed data (UNIQUE constraint)
│   └── build_vector_store.py  # Embed financial documents → JSON vector store
├── data/
│   └── docs/               # Source financial documents for RAG
├── app.py                  # Streamlit frontend — streaming, session state
├── api.py                  # FastAPI backend — POST /analyze
└── requirements.txt
```

---

## What This Demonstrates

- **Multi-agent orchestration** — LangGraph StateGraph with conditional routing and typed shared state
- **RAG pipeline** — custom vector database, embedding generation, cosine similarity retrieval
- **Text2SQL** — LLM-generated SQL with schema injection for accuracy
- **Production deployment** — live Streamlit app, FastAPI backend, streaming output
- **Korean market domain** — Korean public company financial data (DART), Korean language support
- **System design decisions** — documented tradeoffs (custom VectorDB vs. ChromaDB, why schema injection, why streaming)

---

## Related Projects

- **DART MCP Server** — Custom MCP server exposing Korean DART financial data (search, financials, disclosures) as tools for AI agents. [GitHub →](https://github.com/keonhee3337-art)
- **DART Financial App** — Samsung data via DART API → SQLite → RAG → GPT-4o. [Live →](https://keonhee-strategy.streamlit.app)

---

## About

**Keonhee Kim** — Business Administration, Sungkyunkwan University (SKKU), South Korea (UTC+9).

Builds agentic AI systems end-to-end: LangGraph, RAG pipelines, custom VectorDB, Text2SQL, MCP servers, FastAPI, Streamlit. Focused on the intersection of AI engineering and business applications — particularly Korean market data and financial analysis automation.

**Stack:** Python · LangGraph · RAG · OpenAI API · FastAPI · Streamlit · SQLite · NumPy · MCP

[GitHub](https://github.com/keonhee3337-art) · [Live demo](https://keonhee-finagent.streamlit.app)
