import pytest

from nepal_gov_agent.agent import GovAgent


def test_document_qa_returns_result(agent: GovAgent) -> None:
    result = agent.run("What is Nepal's National AI Policy?")
    assert result.answer
    assert result.workflow == "document_qa"
    assert result.confidence in ("high", "medium", "low")
    assert isinstance(result.sources, list)


def test_nepali_query_document_qa(agent: GovAgent) -> None:
    result = agent.run("नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?")
    assert result.answer
    assert len(result.sources) > 0


def test_service_guide_returns_two_steps(agent: GovAgent) -> None:
    result = agent.run("citizenship renewal", workflow="service_guide")
    assert result.workflow == "service_guide"
    assert len(result.steps) == 2
    assert "Eligibility" in result.answer


def test_corpus_search_returns_blocks(agent: GovAgent) -> None:
    result = agent.run("National AI Centre", workflow="corpus_search")
    assert result.workflow == "corpus_search"
    assert len(result.sources) > 0


def test_unknown_workflow_raises(agent: GovAgent) -> None:
    with pytest.raises(ValueError, match="Unknown workflow"):
        agent.run("query", workflow="nonexistent")


def test_session_history_persists(agent: GovAgent) -> None:
    agent.run("What is AI policy?")
    agent.run("What about the Constitution?")
    history = agent.history
    assert len(history) >= 4


def test_clear_history(agent: GovAgent) -> None:
    agent.run("test query")
    agent.clear_history()
    assert agent.history == []
