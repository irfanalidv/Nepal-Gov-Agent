"""
Retrieval benchmark: built-in smoke test plus your own QA pairs
==============================================================
Shows    : ``run_benchmark`` on the real seed corpus (7 hand-curated questions), then
           on a custom QA list, which is how you would track retrieval quality on
           your own documents. Metrics are retrieval-only: Recall@k (expected
           keyword in the top-k blocks), document hit rate, Nepali vs English split.
Data     : the five seed PDFs, downloaded once into ./nepal_gov_data/ (~6.5 MB).
Needs    : ``pip install nepal-gov-agent`` and network access for the first download
           (PDFs + embedding model). No API key.
Run      : python examples/benchmark_corpus.py
"""

import logging
from pathlib import Path

from nepal_gov_agent import GovRAG, download_corpus, run_benchmark

logging.basicConfig(level=logging.WARNING)

rag = GovRAG(corpus_dir=download_corpus())
smoke = run_benchmark(rag, verbose=False)
print(smoke.report())

my_pairs = [
    {
        "query": "Who serves as secretariat of the AI Regulation Council?",
        "expected_keywords": ["National AI Centre"],
        "expected_doc": "National AI Policy-Final_uxc94vg.pdf",
        "language": "english",
    },
    {
        "query": "What is the right to equality?",
        "expected_keywords": ["equal before law", "equal protection"],
        "expected_doc": "Constitution of Nepal (2nd amd. English)_xf33zb3.pdf",
        "language": "english",
    },
]
custom = run_benchmark(rag, qa_pairs=my_pairs, verbose=False, eval_kind="custom")
print(custom.report())

out = Path("benchmark_results.txt")
with out.open("w", encoding="utf-8") as f:
    f.write(smoke.report() + "\n\nPer-query breakdown:\n")
    for r in smoke.per_query:
        status = "hit " if r["keyword_hit"] else "miss"
        f.write(f"{status} [{r['language']}] {r['query']} (recall@3={r['recall@3']})\n")
print(f"\nPer-query results saved to {out}")
