"""
GovAgent document_qa workflow
=============================
Problem  : Government document Q&A requires source tracing — bare LLM answers are unverifiable.
Module   : nepal_gov_agent.agent.GovAgent
Dataset  : Nepal government seed PDFs (downloaded by download_corpus())
Install  : pip install nepal-gov-agent
Env vars : NONE
"""

import logging

from nepal_gov_agent import GovRAG, download_corpus
from nepal_gov_agent.agent import GovAgent

logging.basicConfig(level=logging.INFO)

rag = GovRAG(corpus_dir=download_corpus())
agent = GovAgent(rag=rag, session_id="demo")

result = agent.run("What is the role of the National AI Centre?")
print(result.answer)
print("Confidence:", result.confidence)
print("Sources:", [(s["doc"], s["page"]) for s in result.sources[:3]])
