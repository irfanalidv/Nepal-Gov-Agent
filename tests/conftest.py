import logging
import os
import tempfile

import pytest

from nepal_gov_agent import GovAgent, GovRAG
from nepal_gov_agent.config import GovAgentConfig

logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="session")
def rag() -> GovRAG:
    """One GovRAG index for the full test session (embeddings are expensive)."""
    return GovRAG(corpus_dir="Data/")


@pytest.fixture(scope="module")
def agent(rag: GovRAG) -> GovAgent:
    """One GovAgent per test module sharing the session index; isolated DB file."""
    fd, path = tempfile.mkstemp(suffix="_gov_agent_pytest.db")
    os.close(fd)
    cfg = GovAgentConfig(session_db=path)
    inst = GovAgent(rag=rag, config=cfg, session_id="pytest_agent_module")
    yield inst
    try:
        os.unlink(path)
    except OSError:
        pass
