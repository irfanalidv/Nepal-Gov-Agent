"""
GovRAG — Unified RAG interface for Nepal government documents.

Wires together:
  - RAGNav: hybrid BM25 + vector retrieval with inline citations
  - RAGNav QueryFallback: adaptive retries when retrieval confidence is low

Designed for offline use. No API keys required by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ragnav.answering.inline_citations import CitedAnswer, answer_with_inline_citations
from ragnav.cache.sqlite_cache import (
    EmbeddingCache,
    RetrievalCache,
    SqliteCacheConfig,
    SqliteKV,
)
from ragnav.llm.base import LLMClient
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, ConfidenceLevel
from ragnav.retrieval.fallback import FallbackConfig, QueryFallback
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever

from .config import DEFAULT_RAG_CONFIG, GovRAGConfig
from .ingest import ingest_corpus

logger = logging.getLogger(__name__)


@dataclass
class GovRAGResult:
    answer: str
    sources: list[dict]
    cited_block_ids: tuple[str, ...]
    confidence: str
    query_used: str
    fallback_triggered: bool


class GovRAG:
    def __init__(
        self,
        corpus_dir: str = "Data/",
        config: GovRAGConfig = DEFAULT_RAG_CONFIG,
    ):
        self.config = config
        self.corpus_dir = corpus_dir

        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_kv = SqliteKV(SqliteCacheConfig(db_path=str(cache_dir / "embeddings.db")))
        ret_kv = SqliteKV(SqliteCacheConfig(db_path=str(cache_dir / "retrieval.db")))
        self._emb_cache = EmbeddingCache(emb_kv)
        self._ret_cache = RetrievalCache(ret_kv)

        logger.info("Loading corpus from: %s", corpus_dir)
        self._documents, self._blocks = ingest_corpus(corpus_dir)

        logger.info("Building retrieval index...")

        self._llm: LLMClient = FakeLLMClient()

        self._index = RAGNavIndex.build(
            documents=self._documents,
            blocks=self._blocks,
            llm=self._llm,
            build_vectors=True,
            use_sentence_transformers=True,
            embedding_cache=self._emb_cache,
            embed_batch_size=64,
            vector_model=config.embedding_model,
        )

        self._retriever = RAGNavRetriever(
            index=self._index,
            llm=self._llm,
        )

        logger.info(
            "Ready: %d blocks indexed across %d documents",
            len(self._blocks),
            len(self._documents),
        )

    def ask(
        self,
        query: str,
        *,
        llm: Optional[LLMClient] = None,
        k_final: Optional[int] = None,
        with_citations: bool = True,
    ) -> GovRAGResult:
        cfg = self.config
        k = k_final or cfg.k_final

        retrieval = self._retriever.retrieve(
            query,
            k_bm25=cfg.k_bm25,
            k_vec=cfg.k_vec,
            k_final=k,
            w_bm25=cfg.w_bm25,
            w_vec=cfg.w_vec,
            expand_structure=True,
            retrieval_cache=self._ret_cache,
        )

        fallback_triggered = False
        query_used = query

        if retrieval.confidence == ConfidenceLevel.LOW and cfg.max_fallback_attempts > 1:
            fb = QueryFallback(
                self._retriever,
                self._llm,
                FallbackConfig(
                    max_attempts=cfg.max_fallback_attempts,
                    retry_on={ConfidenceLevel.LOW},
                ),
            )
            fb_result = fb.retrieve(
                query,
                k_bm25=cfg.k_bm25,
                k_vec=cfg.k_vec,
                k_final=k,
                w_bm25=cfg.w_bm25,
                w_vec=cfg.w_vec,
                expand_structure=True,
                retrieval_cache=self._ret_cache,
            )
            if fb_result.improved:
                retrieval = fb_result.final_result
                query_used = fb_result.winning_query
                fallback_triggered = True

        blocks = retrieval.blocks

        sources = []
        for b in blocks:
            text = b.text or ""
            excerpt = text[:200] + "..." if len(text) > 200 else text
            sources.append(
                {
                    "doc": b.doc_id.replace("pdf:", ""),
                    "page": b.anchors.get("page"),
                    "block_id": b.block_id,
                    "heading": " > ".join(b.heading_path) if b.heading_path else None,
                    "excerpt": excerpt,
                }
            )

        gen_llm = llm
        if gen_llm is not None and with_citations and blocks:
            try:
                cited: CitedAnswer = answer_with_inline_citations(
                    llm=gen_llm,
                    query=query,
                    blocks=blocks,
                )
                answer = cited.answer
                cited_ids = cited.cited_block_ids
            except Exception:
                answer = self._simple_answer(query, blocks)
                cited_ids = tuple(b.block_id for b in blocks[:3])
        else:
            answer = self._simple_answer(query, blocks)
            cited_ids = tuple(b.block_id for b in blocks[:3])

        return GovRAGResult(
            answer=answer,
            sources=sources,
            cited_block_ids=cited_ids,
            confidence=retrieval.confidence.value,
            query_used=query_used,
            fallback_triggered=fallback_triggered,
        )

    def _simple_answer(self, query: str, blocks: list[Block]) -> str:
        if not blocks:
            return "No relevant content found in the Nepal government corpus."
        parts = []
        for i, b in enumerate(blocks[:5], 1):
            title = " > ".join(b.heading_path) if b.heading_path else f"Block {i}"
            text = (b.text or "").strip()
            source = b.doc_id.replace("pdf:", "")
            page = b.anchors.get("page", "?")
            parts.append(f"[{source}, p.{page}] {title}\n{text}")
        return "\n\n---\n\n".join(parts)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return self._retriever.retrieve_raw(
            query,
            max_blocks=k,
            k_bm25=self.config.k_bm25,
            k_vec=self.config.k_vec,
            k_final=k,
            w_bm25=self.config.w_bm25,
            w_vec=self.config.w_vec,
            expand_structure=True,
            retrieval_cache=self._ret_cache,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self._documents),
            "blocks": len(self._blocks),
            "corpus_dir": self.corpus_dir,
            "offline": self.config.offline,
        }
