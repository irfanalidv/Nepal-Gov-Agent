"""
Nepal GovAgent quickstart (English + Nepali)
============================================
Problem  : Without cited retrieval, users cannot verify answers against official government PDFs.
Module   : nepal_gov_agent.rag.GovRAG
Dataset  : Nepal government seed PDFs (downloaded by download_corpus())
Install  : pip install nepal-gov-agent
Env vars : NONE
"""

import logging

logging.basicConfig(level=logging.INFO)

from nepal_gov_agent import GovRAG, download_corpus

rag = GovRAG(corpus_dir=download_corpus())

print("Corpus stats:", rag.stats)
print()

result = rag.ask("What is the vision of Nepal's National AI Policy?")
print("Q: What is the vision of Nepal's National AI Policy?")
print("A:", result.answer[:500])
print("Confidence:", result.confidence)
print("Sources:")
for src in result.sources[:3]:
    print(f"  - {src['doc']} page {src['page']}")
print()

result2 = rag.ask("नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?")
print("Q: नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?")
print("A:", result2.answer[:500])
print("Confidence:", result2.confidence)
print("Fallback triggered:", result2.fallback_triggered)
