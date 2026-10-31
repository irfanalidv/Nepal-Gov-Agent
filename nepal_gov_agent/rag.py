"""
GovRAG — Unified RAG interface for Nepal government documents.

Wires together:
  - RAGNav: hybrid BM25 + vector retrieval with inline citations
  - RAGNav QueryFallback: adaptive retries when retrieval confidence is low

Designed for offline use. No API keys required by default.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragnav.answering.inline_citations import CitedAnswer, answer_with_inline_citations
from ragnav.cache.sqlite_cache import EmbeddingCache, SqliteCacheConfig, SqliteKV
from ragnav.llm.base import LLMClient
from ragnav.llm.fake import FakeLLMClient
from ragnav.models import Block, ConfidenceLevel, RetrievalResult
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.retrieval.fallback import FallbackConfig, QueryFallback

from .config import DEFAULT_RAG_CONFIG, GovRAGConfig
from .ingest import ingest_corpus
from .preprocess import preprocess_query

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_MARKER = ".embedding_model"


def _invalidate_embedding_cache_if_model_changed(cache_dir: Path, model: str) -> None:
    """Remove SQLite embedding / retrieval DBs when ``embedding_model`` changes."""
    if not cache_dir.is_dir():
        return
    marker = cache_dir / _EMBEDDING_MODEL_MARKER
    prev: str | None = None
    if marker.is_file():
        try:
            prev = marker.read_text(encoding="utf-8").strip()
        except OSError:
            prev = None
    elif any(cache_dir.glob("*.db")):
        # Upgraded install: had embeddings before marker file existed — rebuild once.
        prev = ""
    else:
        return
    if prev == model:
        return
    logger.warning(
        "Embedding model changed (%r -> %r); clearing cache at %s",
        prev,
        model,
        cache_dir,
    )
    for p in list(cache_dir.iterdir()):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", p, exc)
    marker.unlink(missing_ok=True)


def _write_embedding_model_marker(cache_dir: Path, model: str) -> None:
    (cache_dir / _EMBEDDING_MODEL_MARKER).write_text(model, encoding="utf-8")


def _ranked_blocks(result: RetrievalResult) -> list[Block]:
    """Return ``result.blocks`` with the ranked seed blocks first, in score order.

    With ``expand_structure=True`` RAGNav sorts its output by heading depth and
    block id, i.e. document order. That made ``blocks[0]`` "the earliest page",
    not "the best match", which skewed the extractive answer, the cited ids and
    the benchmark's Recall@1/@3. The trace keeps the ranked seed ids, so restore
    that order here; structural neighbours keep their relative order after the seeds.
    """
    seed_ids = (result.trace or {}).get("seed_block_ids") or []
    if not seed_ids:
        return list(result.blocks)
    rank = {bid: i for i, bid in enumerate(seed_ids)}
    return sorted(result.blocks, key=lambda b: rank.get(b.block_id, len(rank)))


@dataclass
class GovRAGResult:
    """Answer plus provenance returned by :meth:`GovRAG.ask`.

    Attributes:
        answer: Extractive answer (``[doc, p.N] heading`` blocks) or, when an ``llm``
            is passed, an LLM answer with ``[[block_id]]`` inline citations.
        sources: One dict per retrieved block: ``doc``, ``page``, ``block_id``,
            ``heading`` and a 200-character ``excerpt``.
        cited_block_ids: Block ids the answer cites.
        confidence: RAGNav retrieval confidence, ``"high"``, ``"medium"`` or ``"low"``.
        query_used: The (preprocessed) query that produced the final retrieval.
        fallback_triggered: True when a QueryFallback rewrite beat the original query.
    """

    answer: str
    sources: list[dict[str, Any]]
    cited_block_ids: tuple[str, ...]
    confidence: str
    query_used: str
    fallback_triggered: bool


class GovRAG:
    """Hybrid BM25 + dense retrieval with citations over a folder of PDFs.

    Building an instance ingests every ``*.pdf`` in ``corpus_dir``, embeds the
    blocks (cached in SQLite under ``config.cache_dir``) and builds the RAGNav index.

    Args:
        corpus_dir: Flat folder of PDFs (subfolders are not scanned).
        config: Retrieval settings; see :class:`GovRAGConfig`.

    Raises:
        CorpusNotFoundError: ``corpus_dir`` does not exist.
        EmptyCorpusError: no PDFs, or none produced any text.

    Note:
        The underlying SQLite caches are bound to the creating thread; use one
        instance per thread (see ``web/app.py`` for a single-worker pattern).
    """

    def __init__(
        self,
        corpus_dir: str = "Data/",
        config: GovRAGConfig = DEFAULT_RAG_CONFIG,
    ) -> None:
        self.config = config
        self.corpus_dir = corpus_dir

        cache_dir = Path(config.cache_dir)
        _invalidate_embedding_cache_if_model_changed(cache_dir, config.embedding_model)
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_kv = SqliteKV(SqliteCacheConfig(db_path=str(cache_dir / "embeddings.db")))
        self._emb_cache = EmbeddingCache(emb_kv)
        # No RAGNav RetrievalCache: on a cache hit RAGNav reports confidence LOW and
        # drops the ranking trace, so every repeated query looked low-confidence and
        # re-ran QueryFallback. Hybrid search over a cached index is cheap anyway.

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

        _write_embedding_model_marker(cache_dir, config.embedding_model)

        logger.info(
            "Ready: %d blocks indexed across %d documents",
            len(self._blocks),
            len(self._documents),
        )

    def ask(
        self,
        query: str,
        *,
        llm: LLMClient | None = None,
        k_final: int | None = None,
        with_citations: bool = True,
        allowed_doc_ids: set[str] | None = None,
    ) -> GovRAGResult:
        """Retrieve evidence for ``query`` and build an answer.

        Args:
            query: Nepali or English question.
            llm: Optional RAGNav ``LLMClient`` (e.g. :class:`OllamaClient`). Without
                one the answer is extractive and no network call is made.
            k_final: Blocks to keep (default ``config.k_final``).
            with_citations: Ask the LLM for ``[[block_id]]`` inline citations.
            allowed_doc_ids: Restrict retrieval to these RAGNav doc ids
                (``"pdf:<filename>"``).

        If the LLM call fails, the error is logged and the extractive answer is
        returned, so a flaky backend never takes the endpoint down.
        """
        cfg = self.config
        k = k_final or cfg.k_final

        q = preprocess_query(query)

        retrieval = self._retriever.retrieve(
            q,
            k_bm25=cfg.k_bm25,
            k_vec=cfg.k_vec,
            k_final=k,
            w_bm25=cfg.w_bm25,
            w_vec=cfg.w_vec,
            expand_structure=True,
            allowed_doc_ids=allowed_doc_ids,
        )

        fallback_triggered = False
        query_used = q

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
                q,
                k_bm25=cfg.k_bm25,
                k_vec=cfg.k_vec,
                k_final=k,
                w_bm25=cfg.w_bm25,
                w_vec=cfg.w_vec,
                expand_structure=True,
                allowed_doc_ids=allowed_doc_ids,
            )
            if fb_result.improved:
                retrieval = fb_result.final_result
                query_used = preprocess_query(fb_result.winning_query)
                fallback_triggered = True

        blocks = _ranked_blocks(retrieval)

        sources: list[dict[str, Any]] = []
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
                    query=q,
                    blocks=blocks,
                )
                answer = cited.answer
                cited_ids = cited.cited_block_ids
            except Exception as exc:
                # Keep serving an extractive answer, but never fail silently.
                logger.warning(
                    "Cited answer generation failed (%s: %s); returning extractive answer",
                    type(exc).__name__,
                    exc,
                )
                answer = self._simple_answer(q, blocks)
                cited_ids = tuple(b.block_id for b in blocks[:3])
        else:
            answer = self._simple_answer(q, blocks)
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
        """Return the top-``k`` raw blocks, best match first, without building an answer.

        Each dict has ``doc_id``, ``block_id``, ``title``, ``anchors`` and ``content``.
        """
        q = preprocess_query(query)
        result = self._retriever.retrieve(
            q,
            k_bm25=self.config.k_bm25,
            k_vec=self.config.k_vec,
            k_final=k,
            w_bm25=self.config.w_bm25,
            w_vec=self.config.w_vec,
            expand_structure=True,
        )
        # Same dict shape as RAGNavRetriever.retrieve_raw, but in rank order.
        return [
            {
                "doc_id": b.doc_id,
                "block_id": b.block_id,
                "title": " > ".join(b.heading_path) if b.heading_path else None,
                "anchors": b.anchors,
                "content": b.text,
            }
            for b in _ranked_blocks(result)[:k]
        ]

    @property
    def stats(self) -> dict[str, Any]:
        """Document / block counts, corpus folder and the ``offline`` flag."""
        return {
            "documents": len(self._documents),
            "blocks": len(self._blocks),
            "corpus_dir": self.corpus_dir,
            "offline": self.config.offline,
        }
