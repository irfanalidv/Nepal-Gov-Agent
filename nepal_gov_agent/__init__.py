"""
Nepal GovAgent — Agentic AI framework for Nepal's government service layer.

Quick start:
    from nepal_gov_agent import GovRAG

    rag = GovRAG(corpus_dir="Data/")
    result = rag.ask("What is Nepal's National AI Policy?")
    print(result.answer)
    print(result.sources)
"""

from .benchmark import BenchmarkResult, NEPAL_GOV_QA, run_benchmark
from .config import GovRAGConfig
from .rag import GovRAG, GovRAGResult

__version__ = "0.1.0"

__all__ = [
    "GovRAG",
    "GovRAGResult",
    "GovRAGConfig",
    "run_benchmark",
    "NEPAL_GOV_QA",
    "BenchmarkResult",
]
