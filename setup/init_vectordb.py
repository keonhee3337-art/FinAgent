"""
Initialize the custom VectorDB by embedding all financial documents.
Run this once: python setup/init_vectordb.py
Requires OPENAI_API_KEY in .env
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from agent.vector_store import build_vector_store

load_dotenv()

DOCUMENTS = [
    {
        "id": "doc_001",
        "text": """Samsung Electronics Semiconductor Division Analysis 2024:
Samsung's semiconductor business faced severe headwinds in 2023 due to global memory oversupply,
resulting in operating losses exceeding 14 trillion KRW. The company responded by cutting DRAM
and NAND production by approximately 50%. Recovery accelerated in 2024 driven by HBM (High Bandwidth Memory)
demand from AI chip manufacturers including NVIDIA. However, Samsung fell behind SK Hynix in HBM3E
qualification, creating a strategic gap in the AI memory market. Capex discipline improved significantly
with investment focused on advanced nodes (3nm GAA) at Foundry division.""",
        "metadata": {"company": "Samsung Electronics", "topic": "semiconductor", "year": "2024"}
    },
    {
        "id": "doc_002",
        "text": """SK Hynix HBM Market Leadership 2024:
SK Hynix emerged as the dominant supplier of HBM3E memory to NVIDIA, capturing over 70% market share
in the high-margin AI memory segment. This strategic positioning drove a remarkable recovery from
2023's 7.7 trillion KRW operating loss to a 23.5 trillion KRW operating profit in 2024.
The company's early investment in HBM technology — beginning in 2019 — exemplifies how anticipatory
capex allocation creates durable competitive advantage. Revenue grew 102% YoY to 66.2 trillion KRW,
making 2024 the best year in company history.""",
        "metadata": {"company": "SK Hynix", "topic": "HBM", "year": "2024"}
    },
    {
        "id": "doc_003",
        "text": """Korean Semiconductor Industry RAG vs Fine-Tuning Tradeoffs:
For financial analysis applications in the semiconductor industry, RAG (Retrieval-Augmented Generation)
is preferred over fine-tuning because financial data changes quarterly. Fine-tuning encodes knowledge
into model weights — making it stale by the next earnings cycle and requiring expensive retraining.
RAG keeps financial documents external and queryable: update the database, not the model.
The key limitation is chunking strategy: small chunks lose context, large chunks cause the LLM
to drop information buried in the middle (the 'lost in the middle' problem).""",
        "metadata": {"company": "general", "topic": "RAG", "year": "2024"}
    },
    {
        "id": "doc_004",
        "text": """LG Electronics Business Transformation 2022-2024:
LG Electronics completed its exit from the mobile phone business in 2021 and pivoted to three growth
engines: EV components (LG Magna), OLED B2B panels, and home appliance premium positioning.
The EV components JV with Magna International reached 3 trillion KRW in revenue by 2024.
Webos platform in smart TVs generated recurring software/ad revenue, improving margin quality.
Operating profit margin remained stable at ~4% despite raw material pressures, supported by
premiumization strategy and subscription-based appliance business (LG ThinQ UP).""",
        "metadata": {"company": "LG Electronics", "topic": "strategy", "year": "2024"}
    },
    {
        "id": "doc_005",
        "text": """AI in Korean Corporate Strategy - Consulting Perspective:
Korean conglomerates (chaebols) are integrating AI across three layers: (1) Operations — predictive
maintenance, yield optimization in fabs, logistics routing; (2) Products — on-device AI in consumer
electronics, AI-powered home appliances; (3) Business Model — AI-as-a-service offerings.
The primary challenge is data siloing across chaebol subsidiaries, limiting training data quality.
McKinsey estimates Korean manufacturers could unlock 40-60 trillion KRW in value through AI adoption
by 2030. Key bottleneck: shortage of engineers who can bridge business strategy and AI implementation.""",
        "metadata": {"company": "general", "topic": "AI strategy", "year": "2024"}
    },
    {
        "id": "doc_006",
        "text": """Text2SQL Challenges in Financial Database Querying:
Text2SQL systems convert natural language queries into SQL for structured financial databases.
Three primary failure modes: (1) Schema linking failure — the model cannot map 'profit' to the correct
column name like 'operating_profit_billion_krw'; (2) Complex join handling — multi-table queries
requiring JOINs across companies and time periods; (3) SQL hallucination — generating syntactically
invalid SQL or using nonexistent table names. Mitigation: inject the full schema into the prompt,
use few-shot examples, and validate generated SQL before execution. Simple schemas (like a single
financials table) dramatically improve accuracy.""",
        "metadata": {"company": "general", "topic": "Text2SQL", "year": "2024"}
    },
]


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)
    build_vector_store(DOCUMENTS, api_key)
