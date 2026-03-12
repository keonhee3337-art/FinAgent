"""
FinAgent — Streamlit UI
Multi-agent financial analysis — LangGraph + RAG + Text2SQL.

v1.1: Progressive streaming — each agent's output appears as it completes.
      Thread-based session memory via LangGraph checkpointing.
"""

import uuid
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="FinAgent",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def get_graph():
    """Build graph once per process — checkpointer is expensive to initialise."""
    from agent.graph import build_graph
    return build_graph()


# ── Session thread ID (persists across reruns, resets on "New Conversation") ──
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


st.title("FinAgent")
st.caption("Multi-agent financial analysis — LangGraph + RAG + Text2SQL")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Architecture")
    st.markdown("""
    **1. Router Agent**
    - Classifies query: sql_only / rag_only / both

    **2. SQL Agent**
    - Converts query to SQL (Text2SQL)
    - Queries SQLite financial database
    - Companies: Samsung, SK Hynix, LG (2020–2024)

    **3. RAG Agent**
    - Embeds query, searches custom VectorDB (cosine similarity)
    - Retrieves top 3 relevant financial documents

    **4. Report Agent**
    - Synthesizes SQL data + RAG findings into a structured report

    **Orchestration:** LangGraph directs flow. State is shared — each agent reads prior outputs.
    """)
    st.divider()
    st.markdown("**Sample queries:**")
    st.code("Compare Samsung and SK Hynix revenue from 2022 to 2024")
    st.code("Which company had the highest operating profit in 2024?")
    st.code("What drove SK Hynix's recovery in 2024?")
    st.code("Explain the HBM opportunity for Korean chipmakers")
    st.divider()
    st.markdown("**Session memory**")
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ── Main query input ───────────────────────────────────────────────────────────
query = st.text_input(
    "Ask a question about Korean corporate financials:",
    placeholder="e.g. What drove Samsung's revenue decline in 2023?"
)

if st.button("Run Analysis", type="primary") and query:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Create a .env file with your key.")
        st.stop()

    graph = get_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "query": query,
        "sql_result": "",
        "rag_result": "",
        "report": "",
        "route": "",
    }

    # ── Layout: route badge + tabs ─────────────────────────────────────────────
    route_badge = st.empty()
    tab1, tab2, tab3 = st.tabs(["Final Report", "SQL Results", "RAG Findings"])

    with tab2:
        st.subheader("Text2SQL Output")
        sql_placeholder = st.empty()
        sql_placeholder.info("⏳ Waiting for SQL agent...")

    with tab3:
        st.subheader("RAG Output (Custom VectorDB — cosine similarity)")
        rag_placeholder = st.empty()
        rag_placeholder.info("⏳ Waiting for RAG agent...")

    with tab1:
        report_placeholder = st.empty()
        report_placeholder.info("⏳ Waiting for report agent...")

    # ── Stream: each node emits updated state as it completes ─────────────────
    for state in graph.stream(initial_state, config=config, stream_mode="values"):

        # Route badge — appears as soon as router_agent finishes
        if state.get("route"):
            route = state["route"]
            route_labels = {
                "sql_only": ("SQL only", "blue"),
                "rag_only": ("RAG only", "green"),
                "both": ("Both agents", "orange"),
            }
            label_text, label_color = route_labels.get(route, (route, "gray"))
            route_badge.markdown(
                f'<span style="background-color:{label_color};color:white;padding:3px 10px;'
                f'border-radius:4px;font-size:0.8em;font-weight:600;">Route: {label_text}</span>',
                unsafe_allow_html=True
            )

        # SQL result — appears as soon as sql_agent finishes
        if state.get("sql_result"):
            sql_placeholder.code(state["sql_result"], language="text")

        # RAG result — appears as soon as rag_agent finishes
        if state.get("rag_result"):
            rag_placeholder.markdown(state["rag_result"])

        # Report — appears as soon as report_agent finishes
        if state.get("report"):
            report_placeholder.markdown(state["report"])
