"""
Nepal GovAgent — Agentic AI framework for Nepal's government service layer.

Quick start:
    from nepal_gov_agent import GovRAG

    rag = GovRAG(corpus_dir="Data/")
    result = rag.ask("What is Nepal's National AI Policy?")
    print(result.answer)
    print(result.sources)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .agent import GovAgent, GovAgentResult
from .benchmark import (
    NEPAL_GOV_QA,
    BenchmarkResult,
    run_benchmark,
    run_synthetic_benchmark,
)
from .config import GovAgentConfig, GovRAGConfig
from .corpus import download_corpus
from .eval import SyntheticQAPair, generate_synthetic_qa, load_synthetic_qa
from .exceptions import (
    CorpusDownloadError,
    CorpusNotFoundError,
    EmptyCorpusError,
    LLMUnavailableError,
    MissingDependencyError,
    NepalGovAgentError,
)
from .llm_ollama import OllamaClient
from .preprocess import NEPALI_QUESTION_SUFFIXES, preprocess_query
from .rag import GovRAG, GovRAGResult

try:
    __version__ = _pkg_version("nepal-gov-agent")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.6.1"

__all__ = [
    "NEPALI_QUESTION_SUFFIXES",
    "NEPAL_GOV_QA",
    "BenchmarkResult",
    "CorpusDownloadError",
    "CorpusNotFoundError",
    "EmptyCorpusError",
    "GovAgent",
    "GovAgentConfig",
    "GovAgentResult",
    "GovRAG",
    "GovRAGConfig",
    "GovRAGResult",
    "LLMUnavailableError",
    "MissingDependencyError",
    "NepalGovAgentError",
    "OllamaClient",
    "SyntheticQAPair",
    "__version__",
    "download_corpus",
    "generate_synthetic_qa",
    "load_synthetic_qa",
    "preprocess_query",
    "run_benchmark",
    "run_synthetic_benchmark",
]
