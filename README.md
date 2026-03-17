# FinAgent — Multi-Agent Financial Analysis System

**Live demo:** [keonhee-finagent.streamlit.app](https://keonhee-finagent.streamlit.app)  
**Production API:** AWS Lambda — ap-northeast-2 (Seoul)

Built by **Keonhee Kim**, Business Administration student at Sungkyunkwan University (SKKU), South Korea.

---

FinAgent is a production-deployed, multi-agent financial analysis system that automates structured financial data retrieval and document-grounded reasoning for Korean public companies. It answers natural language questions like "Compare Samsung Electronics and SK Hynix operating margins 2020–2024" by routing queries through a dynamic agent pipeline — without any human decision-making at each step.

**What it automates:** The workflow an analyst would manually perform: write SQL, search financial documents, synthesize into a structured report. FinAgent does this end-to-end from a single natural language query — reducing manual analysis from hours to under 2 minutes per company.

---

## Architecture

```
User query (natural language)
    │
    ▼
[Router Agent]
    │  Classifies query type:
    ├── sql_only  → [SQL Agent] ───────────────────────┐
    ├── rag_only  → [RAG Agent] ────────────────────┤
    └── both      → [SQL Agent] → [RAG Agent] ─────────┤
                                                        │
                                                 [Report Agent]
                                                        │
                                              Structured markdown report
                                     (Key Findings | Financial Data | Market Context)
                                                        │
                                              FastAPI backend (POST /analyze)
                                                        │
                                              Streamlit frontend (streaming output)
```

**SQL Agent:** GPT-4o generates SQL with schema injection → executes against SQLite (Samsung Electronics, SK Hynix, LG Electronics, 2020–2024 financials).

**RAG Agent:** Custom vector database query (OpenAI `text-embedding-3-small`, NumPy cosine similarity) → GPT-4o synthesis over retrieved chunks.

**Report Agent:** GPT-4o synthesizes SQL results + RAG findings → structured markdown.

**Orchestration:** LangGraph `StateGraph` with typed `AgentState` (TypedDict). Conditional edges route based on query classification. Streaming via `graph.stream(stream_mode="values")` — output renders progressively as each agent completes.

---

## Key Technical Decisions

### Dynamic routing via Router Agent

A router agent classifies each query before execution, so the pipeline only runs the agents that are needed:

```python
def route_query(state: dict) -> str:
    route = state.get("route", "both")
    if route == "sql_only":
        return "sql_agent"   # numeric data query, no docs needed
    elif route == "rag_only":
        return "rag_agent"   # qualitative question, no SQL needed
    else:
        return "sql_agent"   # both: sql first, then rag
```

This avoids unnecessary LLM calls on simple queries and is the core reason to use LangGraph over a linear pipeline.

### Custom VectorDB (not ChromaDB or Pinecone)

ChromaDB is incompatible with Python 3.14 (Pydantic v1 runtime issue). Rather than downgrade Python, a custom vector database was built from NumPy primitives:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Embeddings (`text-embedding-3-small`, 1536 dims) stored as JSON, retrieved at query time. Zero external VectorDB dependencies.

### LangChain Integration

The custom VectorDB is also wrapped in LangChain's `BaseRetriever` interface (`langchain_rag.py`), making it compatible with any LangChain chain or agent without duplicating the embedding logic:

```python
class FinAgentRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = query_vector_store(query, self.client, top_k=self.top_k)
        return [Document(page_content=r["text"], metadata=r["metadata"]) for r in results]
```

This demonstrates both custom VectorDB implementation and LangChain abstraction — knowing when to build from scratch and when to use the standard interface.

### Text2SQL with schema injection

The SQL agent injects the full SQLite schema into the system prompt, giving GPT-4o exact column names and types:

```python
system_prompt = f"""
You are a SQL expert. The database schema is:
{schema}
Generate a valid SQLite query.
"""
```

This eliminates hallucinated field names and achieves high SQL accuracy on the first attempt.

### LangGraph streaming

Output streams as each agent completes — users see SQL results before RAG finishes:

```python
for state in graph.stream(initial_state, config=config, stream_mode="values"):
    if state.get("route"):      show_route_badge(state["route"])
    if state.get("sql_result"): sql_placeholder.code(state["sql_result"])
    if state.get("rag_result"): rag_placeholder.markdown(state["rag_result"])
    if state.get("report"):     report_placeholder.markdown(state["report"])
```

### Checkpointing with fallback

Uses Postgres (Supabase) if `SUPABASE_DB_URL` is set, falls back to `MemorySaver` otherwise. Each Streamlit session gets a unique `thread_id` — conversation state persists within the session.

---

## Stack

| Layer | Technology |
|-------|------------|
| Agent orchestration | LangGraph `StateGraph` (v1.1+), conditional routing, streaming |
| Structured retrieval | Text2SQL — GPT-4o + SQLite |
| Semantic retrieval | RAG — custom VectorDB (OpenAI embeddings + NumPy cosine similarity) |
| LangChain integration | `BaseRetriever` wrapper (`langchain_rag.py`) — compatible with any LangChain chain |
| Report synthesis | GPT-4o |
| Backend | FastAPI (async, CORS-enabled) |
| Frontend | Streamlit (streaming, session state, `@st.cache_resource`) |
| Cloud | AWS Lambda + API Gateway (ap-northeast-2) |
| Database | SQLite (Korean financial data, 2020–2024) |
| Checkpointing | PostgresSaver (Supabase) with MemorySaver fallback |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |

---

## Data

Korean public company financials sourced from DART (Korea Financial Supervisory Service):

| Company | Years | Fields |
|---------|-------|--------|
| Samsung Electronics (삼성전자) | 2020–2024 | revenue, operating_profit, net_income, total_assets, total_liabilities, equity |
| SK Hynix (SK하이닉스) | 2020–2024 | same |
| LG Electronics (LG전자) | 2020–2024 | same |

All figures in Korean Won (KRW, billions). `UNIQUE(company, year)` constraint prevents duplicate records.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add API key
cp .env.example .env
# Add OPENAI_API_KEY to .env

# 3. Initialize database and build vector store
python setup/init_db.py
python setup/build_vector_store.py

# 4. Run
streamlit run app.py          # Streamlit frontend
uvicorn api:app --reload      # FastAPI backend (alternative)

# 5. LangChain wrapper (optional)
pip install langchain langchain-openai
python langchain_rag.py "What is Samsung's operating profit trend?"
```

---

## Project Structure

```
FinAgent/
├── agent/
│   ├── graph.py            # LangGraph pipeline — StateGraph, conditional edges, streaming
│   ├── router_agent.py     # Query classification: sql_only | rag_only | both
│   ├── sql_agent.py        # Text2SQL — schema injection + GPT-4o + SQLite
│   ├── rag_agent.py        # RAG — custom VectorDB query + GPT-4o synthesis
│   ├── report_agent.py     # Report synthesis — structured markdown output
│   └── vector_store.py     # Custom VectorDB — embeddings + cosine similarity
├── setup/
│   ├── init_db.py          # SQLite schema + seed data (UNIQUE constraint)
│   └── build_vector_store.py  # Embed documents → JSON vector store
├── data/docs/              # Source financial documents for RAG
├── langchain_rag.py        # LangChain BaseRetriever wrapper over custom VectorDB
├── app.py                  # Streamlit — streaming, session state, cached graph
├── api.py                  # FastAPI — POST /analyze
└── requirements.txt
```

---

## What This Demonstrates

- **Multi-agent orchestration** — LangGraph with conditional routing (not just a linear chain)
- **Dynamic query routing** — router agent classifies queries, avoids unnecessary LLM calls
- **RAG pipeline** — custom vector database, cosine similarity, embedding generation
- **LangChain compatibility** — BaseRetriever wrapper; knows when to abstract vs. build from scratch
- **Text2SQL** — schema injection for reliable SQL on first attempt
- **Production deployment** — live Streamlit app + AWS Lambda API, streaming output, per-session state
- **Korean market domain** — Korean corporate financial data (DART), Korean language queries
- **Design decisions documented** — custom VectorDB rationale, routing logic, streaming implementation

---

## Related Projects

- **M&A Due Diligence Suite** — Multi-agent M&A analysis pipeline. LangGraph + DCF valuation + XGBoost distress model + PPTX generator. AWS Lambda live. [GitHub](https://github.com/keonhee3337-art/consulting-emulation)
- **DART MCP Server** — Custom MCP server exposing Korean DART financial data as AI tools. [GitHub](https://github.com/keonhee3337-art/dart-mcp-server)
- **DART Financial App** — Samsung data via DART API → RAG → GPT-4o. [Live](https://keonhee-strategy.streamlit.app)

---

## About

**Keonhee Kim** — Business Administration, Sungkyunkwan University (SKKU), South Korea.

Builds agentic AI systems end-to-end: multi-agent orchestration, RAG pipelines, custom VectorDB, Text2SQL, MCP servers, FastAPI, Streamlit, AWS Lambda. Focus: AI engineering at the intersection of Korean market data and business analysis automation.

**Stack:** Python · LangGraph · LangChain · RAG · OpenAI API · FastAPI · Streamlit · AWS Lambda · SQLite · NumPy · MCP

[GitHub](https://github.com/keonhee3337-art) · [Live demo](https://keonhee-finagent.streamlit.app) · [M&A Suite](https://github.com/keonhee3337-art/consulting-emulation)

---

**Keywords:** `LangGraph` `LangChain` `agentic AI` `multi-agent system` `RAG` `Text2SQL` `custom VectorDB` `Korean market` `DART financial data` `SKKU` `AWS Lambda` `consulting AI` `BCG Gamma` `McKinsey QuantumBlack` `Deloitte AI` `financial analysis automation` `MCP server`
