import pytest

from nepal_gov_agent import GovRAG


def test_bad_corpus_path_raises():
    with pytest.raises(FileNotFoundError):
        GovRAG(corpus_dir="nonexistent_dir_xyz/")


def test_ask_returns_result(rag: GovRAG):
    result = rag.ask("What is Nepal's National AI Policy?")
    assert result.answer
    assert len(result.sources) > 0
    assert result.confidence in ("high", "medium", "low")


def test_ask_nepali_query(rag: GovRAG):
    result = rag.ask("नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?")
    assert result.answer
    assert len(result.sources) > 0


def test_search_returns_blocks(rag: GovRAG):
    blocks = rag.search("National AI Centre", k=3)
    assert len(blocks) >= 1
    assert "content" in blocks[0]


def test_search_key_structure(rag: GovRAG):
    blocks = rag.search("AI policy", k=3)
    if blocks:
        assert "content" in blocks[0]
        assert "doc_id" in blocks[0]
        assert "block_id" in blocks[0]


def test_stats(rag: GovRAG):
    stats = rag.stats
    assert stats["documents"] >= 1
    assert stats["blocks"] >= 3


def test_ask_with_llm_uses_cited_answer(rag: GovRAG):
    from ragnav.llm.fake import FakeLLMClient

    first = rag.search("National AI Centre", k=1)[0]["block_id"]

    class CitingLLM(FakeLLMClient):
        def chat(self, *, messages, model=None, temperature=0):
            # Echoed label form is normalized by ragnav >= 0.5.1.
            return f"The centre coordinates AI work [[BLOCK_ID={first}]]."

    result = rag.ask("National AI Centre", llm=CitingLLM())
    assert result.cited_block_ids == (first,)
    assert f"[[{first}]]" in result.answer


def test_ask_with_failing_llm_falls_back_and_logs(rag: GovRAG, caplog):
    from ragnav.llm.fake import FakeLLMClient

    class BrokenLLM(FakeLLMClient):
        def chat(self, *, messages, model=None, temperature=0):
            raise RuntimeError("rate limited")

    with caplog.at_level("WARNING", logger="nepal_gov_agent.rag"):
        result = rag.ask("National AI Centre", llm=BrokenLLM())
    assert result.answer
    assert "Cited answer generation failed" in caplog.text


def test_search_is_rank_ordered(rag: GovRAG):
    # Regression: RAGNav's structure expansion sorts blocks into document order;
    # GovRAG must hand back the best match first.
    blocks = rag.search("National AI Centre secretariat", k=5)
    assert "secretariat" in blocks[0]["content"]


def test_repeated_ask_is_stable(rag: GovRAG):
    # Regression: a retrieval-cache hit used to report LOW confidence and re-run
    # QueryFallback, flipping fallback_triggered on the second identical call.
    q = "What is the role of the National AI Centre?"
    first = rag.ask(q)
    second = rag.ask(q)
    assert second.confidence == first.confidence
    assert second.fallback_triggered == first.fallback_triggered
    assert [s["block_id"] for s in second.sources] == [s["block_id"] for s in first.sources]


def test_low_confidence_fallback_uses_callers_llm(rag: GovRAG, monkeypatch):
    import dataclasses

    from ragnav.llm.fake import FakeLLMClient
    from ragnav.models import ConfidenceLevel

    real_retrieve = rag._retriever.retrieve
    rewrite = "National AI Centre secretariat"

    def retrieve(query, **kw):
        res = real_retrieve(query, **kw)
        if query != rewrite:  # force the original query to look weak
            res = dataclasses.replace(res, confidence=ConfidenceLevel.LOW, top_score=0.0)
        return res

    monkeypatch.setattr(rag._retriever, "retrieve", retrieve)
    prompts: list[str] = []

    class RewritingLLM(FakeLLMClient):
        def chat(self, *, messages, model=None, temperature=0):
            prompts.append(messages[-1]["content"])
            return rewrite

    result = rag.ask("qqqq zzzz", llm=RewritingLLM(), with_citations=False)
    assert prompts, "QueryFallback should ask the caller's LLM for a rewrite"
    assert result.fallback_triggered
    assert result.query_used == rewrite
    assert "secretariat" in result.answer


def test_weighted_fusion_config_is_used(corpus_dir, tmp_path):
    from conftest import TEST_EMBEDDING_MODEL

    from nepal_gov_agent import GovRAGConfig

    cfg = GovRAGConfig(
        fusion="weighted",
        w_bm25=1.0,
        w_vec=0.0,
        cache_dir=str(tmp_path / "c"),
        embedding_model=TEST_EMBEDDING_MODEL,
    )
    rag = GovRAG(corpus_dir=corpus_dir, config=cfg)
    blocks = rag.search("secretariat", k=1)
    assert "secretariat" in blocks[0]["content"]
