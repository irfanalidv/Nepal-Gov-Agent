"""
GovAgent — AgentEnsemble pipeline over GovRAG with SQLite session memory.

Workflows: document_qa (single ask), service_guide (eligibility then procedure),
corpus_search (raw blocks, no synthesis). Offline; uses GovRAG/FakeLLM only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentensemble.agents.base import BaseAgent
from agentensemble.memory.sqlite_session import SQLiteSession
from agentensemble.orchestration.ensemble import Ensemble

from .config import DEFAULT_AGENT_CONFIG, GovAgentConfig
from .rag import GovRAG, GovRAGResult

logger = logging.getLogger(__name__)


WORKFLOWS: tuple[str, ...] = ("document_qa", "service_guide", "corpus_search")


@dataclass
class GovAgentResult:
    """Outcome of one :meth:`GovAgent.run` call.

    Attributes:
        workflow: Which workflow produced this result.
        answer: Extractive (or LLM-cited) answer text.
        steps: Raw per-step outputs, one dict per pipeline agent.
        sources: Citation dicts (``doc``, ``page``, ``block_id``, ``excerpt`` ...).
        session_id: Session the turn was stored under.
        confidence: ``"high"``, ``"medium"`` or ``"low"``.
    """

    workflow: str
    answer: str
    steps: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    session_id: str
    confidence: str


class _RetrieveAgent(BaseAgent):
    def __init__(
        self,
        rag: GovRAG,
        name: str = "retrieve",
        *,
        rewrite_query: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(name=name, tools=[], max_iterations=1)
        self._rag = rag
        self._rewrite_query = rewrite_query or (lambda q: q)

    def run(self, query: str, **kwargs: Any) -> dict[str, Any]:
        effective = self._rewrite_query(query)
        k = kwargs.get("k_final")
        result: GovRAGResult = self._rag.ask(effective, k_final=k)
        logger.info(
            "step %s: confidence=%s blocks=%d",
            self.name,
            result.confidence,
            len(result.sources),
        )
        return {
            "result": result.answer,
            "metadata": {
                "confidence": result.confidence,
                "sources": result.sources,
                "query_used": result.query_used,
                "fallback_triggered": result.fallback_triggered,
                "step": self.name,
            },
        }

    async def arun(self, query: str, **kwargs: Any) -> dict[str, Any]:
        # GovRAG / SqliteKV are not safe across threads; avoid asyncio.to_thread.
        return self.run(query, **kwargs)


class _SearchAgent(BaseAgent):
    def __init__(self, rag: GovRAG, name: str = "search") -> None:
        super().__init__(name=name, tools=[], max_iterations=1)
        self._rag = rag

    def run(self, query: str, **kwargs: Any) -> dict[str, Any]:
        k = kwargs.get("k", 5)
        blocks = self._rag.search(query, k=k)
        logger.info("step %s: retrieved %d blocks", self.name, len(blocks))
        excerpts: list[str] = []
        for b in blocks:
            c = b.get("content", "") or ""
            excerpts.append(c[:300])
        return {
            "result": "\n\n".join(excerpts),
            "metadata": {
                "blocks": blocks,
                "step": self.name,
            },
        }

    async def arun(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return self.run(query, **kwargs)


class GovAgent:
    """Orchestrates GovRAG retrieval steps and stores turns in SQLite."""

    def __init__(
        self,
        rag: GovRAG,
        config: GovAgentConfig = DEFAULT_AGENT_CONFIG,
        session_id: str = "default",
    ) -> None:
        self._rag = rag
        self._config = config
        self._session = SQLiteSession(
            session_id=session_id,
            db_path=config.session_db,
        )
        logger.info(
            "GovAgent ready conductor=%s session=%s",
            config.conductor,
            session_id,
        )

    def run(
        self,
        query: str,
        *,
        workflow: str = "document_qa",
        k_final: int | None = None,
    ) -> GovAgentResult:
        """Run one workflow and append the user/assistant turn to session memory.

        Args:
            query: Question (Nepali or English) or search text.
            workflow: ``"document_qa"``, ``"service_guide"`` or ``"corpus_search"``.
            k_final: Blocks to keep (QA workflows) or to return (``corpus_search``, default 5).

        Raises:
            ValueError: Unknown ``workflow``. Nothing is written to memory in that case.
        """
        if workflow not in WORKFLOWS:
            raise ValueError(f"Unknown workflow: {workflow!r} (use {', '.join(WORKFLOWS)})")
        self._session.add_messages([{"role": "user", "content": query}])

        if workflow == "document_qa":
            result = self._document_qa(query, k_final=k_final)
        elif workflow == "service_guide":
            result = self._service_guide(query, k_final=k_final)
        else:  # corpus_search
            result = self._corpus_search(query, k=k_final if k_final is not None else 5)

        self._session.add_messages([{"role": "assistant", "content": result.answer}])
        return result

    @property
    def history(self) -> list[dict[str, Any]]:
        """All stored messages for this session, oldest first."""
        return self._session.get_messages()

    def clear_history(self) -> None:
        """Delete every stored message for this session."""
        self._session.clear()

    def _document_qa(self, query: str, *, k_final: int | None) -> GovAgentResult:
        agent = _RetrieveAgent(self._rag, name="qa")
        step_result = agent.run(query, k_final=k_final)
        meta = step_result["metadata"]

        return GovAgentResult(
            workflow="document_qa",
            answer=step_result["result"],
            steps=[step_result],
            sources=list(meta.get("sources", [])),
            session_id=self._session.session_id,
            confidence=str(meta.get("confidence", "low")),
        )

    def _service_guide(self, query: str, *, k_final: int | None) -> GovAgentResult:
        agents: dict[str, BaseAgent] = {
            "eligibility": _RetrieveAgent(
                self._rag,
                name="eligibility",
                rewrite_query=lambda q: "Who is eligible for: " + q,
            ),
            "procedure": _RetrieveAgent(
                self._rag,
                name="procedure",
                rewrite_query=lambda q: "What are the steps and documents required for: " + q,
            ),
        }

        pipeline = Ensemble(agents=agents, conductor="pipeline")
        raw = pipeline.perform(query, k_final=k_final)
        steps = list(raw.get("results", {}).values())
        all_sources: list[dict[str, Any]] = []
        seen_block_ids: set[str] = set()
        for s in steps:
            for src in s.get("metadata", {}).get("sources", []):
                bid = str(src.get("block_id", ""))
                if bid:
                    if bid in seen_block_ids:
                        continue
                    seen_block_ids.add(bid)
                all_sources.append(src)

        eligibility_answer = steps[0]["result"] if steps else ""
        procedure_answer = steps[1]["result"] if len(steps) > 1 else ""

        answer = "**Eligibility**\n" + eligibility_answer + "\n\n**Steps**\n" + procedure_answer

        final_conf = str(steps[-1].get("metadata", {}).get("confidence", "low")) if steps else "low"

        return GovAgentResult(
            workflow="service_guide",
            answer=answer,
            steps=steps,
            sources=all_sources,
            session_id=self._session.session_id,
            confidence=final_conf,
        )

    def _corpus_search(self, query: str, *, k: int) -> GovAgentResult:
        agent = _SearchAgent(self._rag, name="search")
        step_result = agent.run(query, k=k)
        blocks = step_result["metadata"].get("blocks", [])

        sources = [
            {
                "doc": str(b.get("doc_id", "")).replace("pdf:", ""),
                "page": (b.get("anchors") or {}).get("page"),
                "block_id": b.get("block_id", ""),
                "excerpt": (b.get("content", "") or "")[:200],
            }
            for b in blocks
        ]

        return GovAgentResult(
            workflow="corpus_search",
            answer=step_result["result"],
            steps=[step_result],
            sources=sources,
            session_id=self._session.session_id,
            confidence="medium" if blocks else "low",
        )
