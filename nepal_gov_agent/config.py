"""
Default configuration tuned for Nepal government corpus.

Key decisions:
- Hybrid BM25 + dense retrieval fused with Reciprocal Rank Fusion (RRF, k=60),
  RAGNav's default. Under RRF only rank positions matter, so ``w_bm25`` / ``w_vec``
  act as on/off switches (0 disables that retriever). Set ``fusion="weighted"`` to
  blend min-max normalised scores with those weights instead.
- Offline by default: sentence-transformers + optional local Ollama for answers.
- Default embedding model: intfloat/multilingual-e5-small (100+ languages).
- k_final=8: enough context without overloading the answer LLM context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class GovRAGConfig:
    """Retrieval settings for :class:`~nepal_gov_agent.GovRAG` (immutable)."""

    # Retrieval weights. Used as score weights only when fusion="weighted";
    # under the default RRF fusion a weight of 0 disables that retriever.
    w_bm25: float = 0.5
    w_vec: float = 0.5
    k_bm25: int = 30
    k_vec: int = 30
    k_final: int = 8

    # "rrf" (reciprocal rank fusion, default) or "weighted" (min-max score blend)
    fusion: Literal["rrf", "weighted"] = "rrf"

    # Embedding model — offline, multilingual (includes Nepali)
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Fallback config (RAGNav QueryFallback)
    max_fallback_attempts: int = 3

    # Cache
    cache_dir: str = ".nepal_gov_cache"

    # Informational: GovRAG itself never calls a remote API (reported by .stats)
    offline: bool = True

    # Reserved; not used by retrieval yet (queries are language-agnostic)
    language: str = "auto"  # "nepali", "english", or "auto"

    def __post_init__(self) -> None:
        if self.fusion not in ("rrf", "weighted"):
            raise ValueError(f"fusion must be 'rrf' or 'weighted', got {self.fusion!r}")
        if self.w_bm25 < 0 or self.w_vec < 0 or self.w_bm25 + self.w_vec == 0:
            raise ValueError("w_bm25 and w_vec must be >= 0 and not both 0")
        if min(self.k_bm25, self.k_vec, self.k_final) < 1:
            raise ValueError("k_bm25, k_vec and k_final must be >= 1")


@dataclass(frozen=True)
class GovAgentConfig:
    """Settings for :class:`~nepal_gov_agent.GovAgent`.

    Only ``session_db`` changes behaviour today; ``conductor``, ``max_iterations`` and
    ``rag`` are kept for API compatibility (workflows always use the pipeline conductor).
    """

    # Agent orchestration (informational; see docstring)
    conductor: str = "pipeline"  # supervisor | pipeline | swarm
    max_iterations: int = 5

    # Memory
    session_db: str = "gov_agent_sessions.db"

    # RAG config
    rag: GovRAGConfig = field(default_factory=GovRAGConfig)


DEFAULT_RAG_CONFIG = GovRAGConfig()
DEFAULT_AGENT_CONFIG = GovAgentConfig()
