import logging
import os
import shutil
import tempfile

import pytest

from nepal_gov_agent import GovAgent, GovRAG
from nepal_gov_agent.config import GovAgentConfig, GovRAGConfig

logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="session")
def rag() -> GovRAG:
    """One GovRAG index for the full test session (embeddings are expensive)."""
    tmp = tempfile.mkdtemp(prefix="nepal_gov_pytest_")
    cfg = GovRAGConfig(cache_dir=os.path.join(tmp, ".nepal_gov_cache"))
    try:
        yield GovRAG(corpus_dir="Data/", config=cfg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
