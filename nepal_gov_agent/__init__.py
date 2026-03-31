"""
Nepal GovAgent — Agentic AI framework for Nepal's government service layer.

Quick start:
    from nepal_gov_agent import GovRAG

    rag = GovRAG(corpus_dir="Data/")
    result = rag.ask("What is Nepal's National AI Policy?")
    print(result.answer)
    print(result.sources)
"""

from .agent import GovAgent, GovAgentResult
from .benchmark import BenchmarkResult, NEPAL_GOV_QA, run_benchmark
from .config import GovRAGConfig
from .corpus import download_corpus
from .llm_ollama import OllamaClient
from .preprocess import NEPALI_QUESTION_SUFFIXES, preprocess_query
from .rag import GovRAG, GovRAGResult

__version__ = "0.3.0"

__all__ = [
    "GovRAG",
    "GovRAGResult",
    "GovRAGConfig",
    "GovAgent",
    "GovAgentResult",
    "OllamaClient",
    "preprocess_query",
    "NEPALI_QUESTION_SUFFIXES",
    "download_corpus",
    "run_benchmark",
    "NEPAL_GOV_QA",
    "BenchmarkResult",
]
