"""
LangGraph orchestration — defines the multi-agent pipeline as a directed graph.

Flow: SQL Agent → RAG Agent → Report Agent → END

Each node is a function that takes the full state dict and returns an updated state dict.
State passes between agents — each agent adds its output key and forwards everything downstream.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from agent.sql_agent import run_sql_agent
from agent.rag_agent import run_rag_agent
from agent.report_agent import run_report_agent


class AgentState(TypedDict):
    """Shared state object that flows through the entire pipeline."""
    query: str           # Original user query
    sql_result: str      # Output from Text2SQL agent
    rag_result: str      # Output from RAG agent
    report: str          # Final synthesized report


def build_graph():
    """Build and compile the LangGraph multi-agent pipeline."""
    workflow = StateGraph(AgentState)

    # Register nodes (agents)
    workflow.add_node("sql_agent", run_sql_agent)
    workflow.add_node("rag_agent", run_rag_agent)
    workflow.add_node("report_agent", run_report_agent)

    # Define edges (execution order)
    workflow.set_entry_point("sql_agent")
    workflow.add_edge("sql_agent", "rag_agent")
    workflow.add_edge("rag_agent", "report_agent")
    workflow.add_edge("report_agent", END)

    return workflow.compile()
