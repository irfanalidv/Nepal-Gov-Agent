"""
Default configuration tuned for Nepal government corpus.

Key decisions:
- Balanced BM25 / vector weights (0.5 / 0.5) with multilingual embeddings
  so Nepali queries benefit from dense retrieval as well as keywords.
- Offline by default: sentence-transformers + optional local Ollama for answers.
- Default embedding model: intfloat/multilingual-e5-small (100+ languages).
- k_final=8: enough context without overloading the answer LLM context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GovRAGConfig:
    # Retrieval weights — balanced with multilingual embeddings (Nepali + English)
    w_bm25: float = 0.5
    w_vec: float = 0.5
    k_bm25: int = 30
    k_vec: int = 30
    k_final: int = 8

    # Embedding model — offline, multilingual (includes Nepali)
    embedding_model: str = "intfloat/multilingual-e5-small"

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
