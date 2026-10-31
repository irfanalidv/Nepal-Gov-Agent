"""Shared fixtures.

The seed PDFs are not stored in git. By default the suite builds a small synthetic
English corpus with PyMuPDF so it runs offline and fast. To run against the real
seed corpus instead::

    python -c "from nepal_gov_agent import download_corpus; download_corpus('nepal_gov_data')"
    NEPAL_GOV_CORPUS_DIR=nepal_gov_data pytest

``NEPAL_GOV_TEST_EMBEDDING_MODEL`` overrides the (small) embedding model used in tests.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import pymupdf
import pytest

from nepal_gov_agent import GovAgent, GovRAG
from nepal_gov_agent.config import GovAgentConfig, GovRAGConfig

logging.basicConfig(level=logging.WARNING)

TEST_EMBEDDING_MODEL = os.environ.get(
    "NEPAL_GOV_TEST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Short paraphrases of the kinds of passages found in the seed corpus.
_SYNTHETIC_DOCS: dict[str, list[str]] = {
    "National AI Policy-Final_uxc94vg.pdf": [
        "National Artificial Intelligence Policy. Vision: an ethical, human-centric, "
        "inclusive and sustainable use of AI for national development.",
        "The National AI Centre shall act as the secretariat for coordination, research and "
        "regulation of artificial intelligence. The policy aims to train 5000 AI "
        "professionals within five years.",
        "AI readiness rests on four pillars: data, infrastructure, policy and human resources.",
    ],
    "Constitution of Nepal (2nd amd. English)_xf33zb3.pdf": [
        "Part 3 Fundamental Rights and Duties. Every citizen shall have the right to equality "
        "before the law. No citizen shall be discriminated against.",
        "Right to citizenship. Citizenship certificates may be renewed by applying to the "
        "District Administration Office with the required documents.",
    ],
    "dnf_jbji8eb.pdf": [
        "Digital Nepal Framework 2.0 sets out a digital transformation roadmap across eight "
        "sectors including agriculture, health, education and finance.",
    ],
}


def _build_synthetic_corpus(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, pages in _SYNTHETIC_DOCS.items():
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
        doc.save(str(dest / name))
        doc.close()


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    env_dir = os.environ.get("NEPAL_GOV_CORPUS_DIR")
    if env_dir:
        return env_dir
    path = tmp_path_factory.mktemp("corpus")
    _build_synthetic_corpus(path)
    return str(path)


@pytest.fixture(scope="session")
def rag(corpus_dir: str) -> GovRAG:
    """One GovRAG index for the full test session (embeddings are expensive)."""
    tmp = tempfile.mkdtemp(prefix="nepal_gov_pytest_")
    cfg = GovRAGConfig(
        cache_dir=os.path.join(tmp, ".nepal_gov_cache"),
        embedding_model=TEST_EMBEDDING_MODEL,
    )
    try:
        yield GovRAG(corpus_dir=corpus_dir, config=cfg)
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
