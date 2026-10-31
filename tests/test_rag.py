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
