import pytest

from nepal_gov_agent import GovRAG


@pytest.fixture(scope="session")
def rag() -> GovRAG:
    """One GovRAG index for the full test session (embeddings are expensive)."""
    return GovRAG(corpus_dir="Data/", verbose=False)
