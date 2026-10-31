"""
Quickstart: ask the real Nepal government seed corpus (English + Nepali)
=======================================================================
Shows    : download the five seed PDFs, build the hybrid index, ask one English and
           one Nepali question, print the extractive answer, confidence and the
           (document, page) citations you can check against the PDF.
Data     : the five seed PDFs, downloaded once into ./nepal_gov_data/ (~6.5 MB).
Needs    : ``pip install nepal-gov-agent`` and network access for the first run
           (PDFs + embedding model). No API key.
Run      : python examples/quickstart.py
"""

import logging

from nepal_gov_agent import GovRAG, download_corpus

logging.basicConfig(level=logging.WARNING)

rag = GovRAG(corpus_dir=download_corpus())
print("Corpus stats:", rag.stats)

for question in (
    "What is the role of the National AI Centre?",
    "नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?",
):
    result = rag.ask(question)
    print(f"\nQ: {question}")
    print("A:", " ".join(result.answer.split())[:300], "...")
    print("Confidence:", result.confidence, "| fallback:", result.fallback_triggered)
    for src in result.sources[:3]:
        print(f"  - {src['doc']} page {src['page']}")
