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
from .config import DEFAULT_RAG_CONFIG, GovAgentConfig, GovRAGConfig
from .rag import GovRAG, GovRAGResult

__version__ = "0.1.0"
__author__ = "Irfan Ali"

__all__ = [
    "GovRAG",
    "GovRAGResult",
    "GovRAGConfig",
    "GovAgentConfig",
    "DEFAULT_RAG_CONFIG",
    "run_benchmark",
    "NEPAL_GOV_QA",
    "BenchmarkResult",
    "__version__",
]
