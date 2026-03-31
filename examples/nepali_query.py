"""
Nepali query against government corpus
======================================
Problem  : Nepali questions must retrieve the same policy PDFs as English without a separate stack.
Module   : nepal_gov_agent.rag.GovRAG
Dataset  : Nepal government PDFs (Data/ folder — included in repo)
Install  : pip install nepal-gov-agent
Env vars : NONE
"""

import logging

logging.basicConfig(level=logging.INFO)

from nepal_gov_agent import GovRAG

rag = GovRAG(corpus_dir="Data/")
r = rag.ask("नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?")
print(r.answer[:1200])
print("\nConfidence:", r.confidence)
