"""
FinAgent — Streamlit UI
A multi-agent financial intelligence system built on LangGraph + RAG + Text2SQL.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from agent.graph import build_graph

load_dotenv()

st.set_page_config(
    page_title="FinAgent",
    page_icon="📊",
    layout="wide"
)

st.title("FinAgent")
st.caption("Multi-agent financial analysis — LangGraph + RAG + Text2SQL")

# Sidebar: architecture overview
with st.sidebar:
    st.header("Pipeline Architecture")
    st.markdown("""
    **1. SQL Agent**
    - Converts your query to SQL (Text2SQL)
    - Queries SQLite financial database
    - Companies: Samsung, SK Hynix, LG (2020–2024)

    **2. RAG Agent**
    - Embeds your query and searches custom VectorDB (cosine similarity)
    - Retrieves top 3 relevant financial documents
    - Generates grounded answer from retrieved context

    **3. Report Agent**
    - Synthesizes SQL data + RAG findings
    - Outputs a structured analyst report

    **Orchestration:** LangGraph directs the flow between agents.
    State is shared — each agent reads prior outputs and adds its own.
    """)
    st.divider()
    st.markdown("**Sample queries:**")
    st.code("Compare Samsung and SK Hynix revenue from 2022 to 2024")
    st.code("Which company had the highest operating profit in 2024?")
    st.code("What drove SK Hynix's recovery in 2024?")
    st.code("Explain the HBM opportunity for Korean chipmakers")

# Main query input
query = st.text_input(
    "Ask a question about Korean corporate financials:",
    placeholder="e.g. What drove Samsung's revenue decline in 2023?"
)

if st.button("Run Analysis", type="primary") and query:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Create a .env file with your key.")
        st.stop()

    with st.spinner("Running multi-agent pipeline..."):
        graph = build_graph()
        result = graph.invoke({"query": query, "sql_result": "", "rag_result": "", "report": ""})

    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["Final Report", "SQL Results", "RAG Findings"])

    with tab1:
        st.markdown(result["report"])

    with tab2:
        st.subheader("Text2SQL Output")
        st.code(result["sql_result"], language="text")

    with tab3:
        st.subheader("RAG Output (Custom VectorDB — cosine similarity)")
        st.markdown(result["rag_result"])
