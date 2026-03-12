"""
FinAgent — FastAPI backend
Exposes the LangGraph pipeline as a REST API endpoint.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from functools import lru_cache
from agent.graph import build_graph


@lru_cache(maxsize=1)
def get_graph():
    """Build the LangGraph pipeline once and cache it for all requests."""
    return build_graph()

load_dotenv()

app = FastAPI(
    title="FinAgent API",
    description="Multi-agent financial analysis — LangGraph + RAG + Text2SQL",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str


class AnalysisResponse(BaseModel):
    query: str
    sql_result: str
    rag_result: str
    report: str


@app.get("/")
def root():
    return {"status": "ok", "service": "FinAgent API"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: QueryRequest):
    """Run the full multi-agent pipeline on a financial query."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    graph = get_graph()
    result = graph.invoke({
        "query": request.query,
        "sql_result": "",
        "rag_result": "",
        "report": ""
    })

    return AnalysisResponse(**result)


@app.get("/health")
def health():
    return {"status": "healthy", "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))}
