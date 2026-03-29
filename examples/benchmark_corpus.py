"""
Run the Nepal GovAgent benchmark.

Measures:
- Recall@1, @3, @5 — is the right document retrieved?
- Keyword hit rate — are expected terms in retrieved blocks?
- Nepali vs English split — language gap analysis

Run:
    python examples/benchmark_corpus.py
"""

from nepal_gov_agent import GovRAG, run_benchmark

rag = GovRAG(corpus_dir="Data/")
results = run_benchmark(rag, verbose=True)

with open("benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write(results.report())
    f.write("\n\nPer-query breakdown:\n")
    for r in results.per_query:
        status = "✓" if r["keyword_hit"] else "✗"
        f.write(f"{status} [{r['language']}] {r['query']} (recall@3={r['recall@3']})\n")

print("\nResults saved to benchmark_results.txt")
