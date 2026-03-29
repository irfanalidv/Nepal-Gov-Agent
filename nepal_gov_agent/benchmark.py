"""
Nepal GovAgent benchmark harness.

Uses actual Nepal government QA pairs to measure retrieval quality.
Metrics: Recall@k, keyword hit rate, document hit rate.

Real benchmark results from underlying libraries (on standard datasets):
  SQuAD:  Hybrid RAGNav R@1=0.864, R@3=0.956, R@5=0.978, MRR@10=0.912
  CUAD:   Hybrid R@3=0.071 (legal contracts — harder, fewer training signals)

Nepal gov corpus benchmark will differ — run this to get your numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .rag import GovRAG


NEPAL_GOV_QA = [
    {
        "query": "What is the vision of Nepal's National AI Policy?",
        "expected_keywords": ["ethical", "human-centric", "inclusive", "sustainable"],
        "expected_doc": "National AI Policy-Final_uxc94vg.pdf",
        "language": "english",
    },
    {
        "query": "What are the four pillars of Nepal's AI readiness?",
        "expected_keywords": ["data", "infrastructure", "policy", "resources"],
        "expected_doc": "National AI Policy-Final_uxc94vg.pdf",
        "language": "english",
    },
    {
        "query": "What is the role of the National AI Centre?",
        "expected_keywords": ["secretariat", "regulation", "research", "coordination"],
        "expected_doc": "National AI Policy-Final_uxc94vg.pdf",
        "language": "english",
    },
    {
        "query": "How many AI professionals does Nepal aim to train?",
        "expected_keywords": ["5000", "five years"],
        "expected_doc": "National AI Policy-Final_uxc94vg.pdf",
        "language": "english",
    },
    {
        "query": "What does the Constitution say about fundamental rights?",
        "expected_keywords": ["right", "citizen", "equality"],
        "expected_doc": "Constitution of Nepal (2nd amd. English)_xf33zb3.pdf",
        "language": "english",
    },
    {
        "query": "What is Digital Nepal Framework 2.0?",
        "expected_keywords": ["digital", "transformation", "framework", "sectors"],
        "expected_doc": "dnf_jbji8eb.pdf",
        "language": "english",
    },
    # Nepali query; National AI Policy PDF text is largely English. English keywords
    # keep keyword-hit metrics meaningful (retrieval), not false zeros from language mismatch.
    {
        "query": "नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?",
        "expected_keywords": ["ethical", "inclusive", "development"],
        "expected_doc": "National AI Policy-Final_uxc94vg.pdf",
        "language": "nepali",
    },
]


@dataclass
class BenchmarkResult:
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    keyword_hit_rate: float
    doc_hit_rate: float
    nepali_recall: float
    english_recall: float
    per_query: list[dict]

    def report(self) -> str:
        lines = [
            "=" * 60,
            "Nepal GovAgent Benchmark Results",
            "=" * 60,
            f"Total queries:      {self.total_queries}",
            f"Recall@1:           {self.recall_at_1:.3f}",
            f"Recall@3:           {self.recall_at_3:.3f}",
            f"Recall@5:           {self.recall_at_5:.3f}",
            f"Keyword hit rate:   {self.keyword_hit_rate:.3f}",
            f"Doc hit rate:       {self.doc_hit_rate:.3f}",
            f"Nepali recall@3:    {self.nepali_recall:.3f}",
            f"English recall@3:   {self.english_recall:.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def run_benchmark(
    rag: GovRAG,
    qa_pairs: Optional[list[dict]] = None,
    k_values: Optional[list[int]] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run retrieval benchmark against Nepal gov QA pairs.

    Args:
        rag: Initialized GovRAG instance
        qa_pairs: List of QA dicts (default: NEPAL_GOV_QA)
        k_values: Recall@k values to compute
        verbose: Print per-query results

    Returns:
        BenchmarkResult with all metrics
    """
    pairs = qa_pairs or NEPAL_GOV_QA
    k_values = k_values or [1, 3, 5]

    per_query: list[dict] = []
    keyword_hits: list[float] = []
    doc_hits: list[float] = []
    recall_at: dict[int, list[float]] = {k: [] for k in k_values}
    nepali_recall: list[float] = []
    english_recall: list[float] = []

    max_k = max(k_values)

    for qa in pairs:
        query = qa["query"]
        expected_keywords = [kw.lower() for kw in qa.get("expected_keywords", [])]
        expected_doc = qa.get("expected_doc", "")
        lang = qa.get("language", "english")

        blocks = rag.search(query, k=max_k)
        retrieved_texts = [b.get("content", "").lower() for b in blocks]
        retrieved_docs = [
            (b.get("doc_id") or "").replace("pdf:", "", 1) for b in blocks
        ]

        if expected_keywords:
            kw_hit = any(kw in text for kw in expected_keywords for text in retrieved_texts)
        else:
            kw_hit = False
        keyword_hits.append(float(kw_hit))

        doc_hit = any(expected_doc in doc_id for doc_id in retrieved_docs)
        doc_hits.append(float(doc_hit))

        for kk in k_values:
            top_k_texts = retrieved_texts[:kk]
            if expected_keywords:
                hit = any(kw in text for kw in expected_keywords for text in top_k_texts)
            else:
                hit = False
            recall_at[kk].append(float(hit))

        r3 = recall_at[3][-1] if 3 in recall_at and recall_at[3] else 0.0
        if lang == "nepali":
            nepali_recall.append(r3)
        else:
            english_recall.append(r3)

        result = {
            "query": query[:60] + "..." if len(query) > 60 else query,
            "language": lang,
            "keyword_hit": kw_hit,
            "doc_hit": doc_hit,
            "recall@3": r3,
        }
        per_query.append(result)

        if verbose:
            status = "✓" if kw_hit else "✗"
            print(f"  {status} [{lang}] {query[:50]}...")

    def mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    out = BenchmarkResult(
        total_queries=len(pairs),
        recall_at_1=mean(recall_at.get(1, [])),
        recall_at_3=mean(recall_at.get(3, [])),
        recall_at_5=mean(recall_at.get(5, [])),
        keyword_hit_rate=mean(keyword_hits),
        doc_hit_rate=mean(doc_hits),
        nepali_recall=mean(nepali_recall),
        english_recall=mean(english_recall),
        per_query=per_query,
    )

    if verbose:
        print("\n" + out.report())

    return out
