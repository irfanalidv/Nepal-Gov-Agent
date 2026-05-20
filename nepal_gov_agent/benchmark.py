"""
Nepal GovAgent benchmark harness.

Two modes:
  1. Hand-curated smoke test (NEPAL_GOV_QA, 7 questions)
       - Lives in this file, easy to read, intentionally small.
       - Useful for "is anything fundamentally broken" checks.
  2. Synthetic eval (loaded from eval_data/synthetic_qa_v1.jsonl)
       - LLM-generated, validated for keyword presence at generation time.
       - NOT human-validated. Reports are clearly labeled.

Metrics: Recall@k, keyword hit rate, document hit rate, per-language breakdown.

Reference numbers from underlying libraries on standard datasets:
  SQuAD:  Hybrid RAGNav R@1=0.864, R@3=0.956, R@5=0.978, MRR@10=0.912
  CUAD:   Hybrid R@3=0.071 (legal contracts — harder, fewer training signals)

The Nepal gov corpus is its own beast — run the benchmark to get numbers
that actually apply to your install.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .rag import GovRAG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hand-curated smoke test
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

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
    # Provenance — what did we actually evaluate against?
    eval_kind: str = "smoke_test"   # "smoke_test" | "synthetic" | "custom"
    is_synthetic: bool = False
    generator_model: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        header_label = {
            "smoke_test": "Hand-curated smoke test (7 questions)",
            "synthetic":  "SYNTHETIC eval — LLM-generated, NOT human-validated",
            "custom":     "Custom eval set",
        }.get(self.eval_kind, self.eval_kind)

        lines = [
            "=" * 70,
            "Nepal GovAgent Benchmark Results",
            f"Eval set: {header_label}",
        ]
        if self.is_synthetic and self.generator_model:
            lines.append(f"Generator: {self.generator_model}")
        lines += [
            "=" * 70,
            f"Total queries:      {self.total_queries}",
            f"Recall@1:           {self.recall_at_1:.3f}",
            f"Recall@3:           {self.recall_at_3:.3f}",
            f"Recall@5:           {self.recall_at_5:.3f}",
            f"Keyword hit rate:   {self.keyword_hit_rate:.3f}",
            f"Doc hit rate:       {self.doc_hit_rate:.3f}",
            f"Nepali recall@3:    {self.nepali_recall:.3f}",
            f"English recall@3:   {self.english_recall:.3f}",
            "=" * 70,
        ]
        if self.is_synthetic:
            lines += [
                "",
                "⚠️  These numbers come from an LLM-generated eval set. Use them",
                "    as a regression signal, not as ground truth. See",
                "    eval_data/README.md and docs/synthetic-eval.md.",
            ]
        for note in self.notes:
            lines.append(f"  - {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    rag: GovRAG,
    qa_pairs: Optional[list[dict]] = None,
    k_values: Optional[list[int]] = None,
    verbose: bool = True,
    *,
    eval_kind: str = "smoke_test",
    is_synthetic: bool = False,
    generator_model: Optional[str] = None,
) -> BenchmarkResult:
    """
    Run retrieval benchmark against a list of QA pairs.

    Args:
        rag:        Initialized GovRAG instance.
        qa_pairs:   List of QA dicts. Defaults to ``NEPAL_GOV_QA`` (smoke test).
        k_values:   Recall@k values to compute (default [1, 3, 5]).
        verbose:    Log per-query results.
        eval_kind:  "smoke_test" | "synthetic" | "custom" — labels the report.
        is_synthetic: True if pairs were LLM-generated.
        generator_model: Model name to record in the report header.
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
            logger.info("  %s [%s] %.50s...", status, lang, query)

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
        eval_kind=eval_kind,
        is_synthetic=is_synthetic,
        generator_model=generator_model,
    )

    if verbose:
        logger.info("\n%s", out.report())

    return out


def run_synthetic_benchmark(
    rag: GovRAG,
    path: str = "eval_data/synthetic_qa_v1.jsonl",
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Convenience wrapper that loads the synthetic QA file and runs the benchmark
    with the right honesty flags set.
    """
    from .eval.synthetic import load_synthetic_qa, to_benchmark_format

    pairs = load_synthetic_qa(path)
    if not pairs:
        raise ValueError(f"No pairs loaded from {path}")
    model = pairs[0].generator_model
    return run_benchmark(
        rag,
        qa_pairs=to_benchmark_format(pairs),
        verbose=verbose,
        eval_kind="synthetic",
        is_synthetic=True,
        generator_model=model,
    )
