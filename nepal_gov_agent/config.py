"""
Default configuration tuned for Nepal government corpus.

Key decisions:
- BM25 weight higher (0.6) than vector (0.4): Nepal gov docs have specific
  legal terminology that keyword search handles better than embeddings.
- Offline=True by default: no cloud API assumptions.
- Sentence-transformer model: all-MiniLM-L6-v2 — small, fast, CPU-runnable.
- k_final=8: enough context without overloading LLM context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GovRAGConfig:
    # Retrieval weights — BM25 heavier for legal/gov terminology
    w_bm25: float = 0.6
    w_vec: float = 0.4
    k_bm25: int = 30
    k_vec: int = 30
    k_final: int = 8

    # Embedding model — offline, CPU-runnable
    embedding_model: str = "all-MiniLM-L6-v2"

    # Fallback config (RAGNav QueryFallback)
    max_fallback_attempts: int = 3

    # Cache
    cache_dir: str = ".nepal_gov_cache"

    # Offline mode — no external API calls
    offline: bool = True

    # Language handling
    language: str = "auto"  # "nepali", "english", or "auto"


@dataclass(frozen=True)
class GovAgentConfig:
    # Agent orchestration
    conductor: str = "pipeline"  # supervisor | pipeline | swarm
    max_iterations: int = 5

    # Memory
    session_db: str = "gov_agent_sessions.db"

    # RAG config
    rag: GovRAGConfig = field(default_factory=GovRAGConfig)


DEFAULT_RAG_CONFIG = GovRAGConfig()
DEFAULT_AGENT_CONFIG = GovAgentConfig()
