from nepal_gov_agent import GovRAG


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
    assert stats["blocks"] >= 10
