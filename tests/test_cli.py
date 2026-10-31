"""CLI sub-commands, run against the session-scoped test index (no network)."""

from __future__ import annotations

import pytest

from nepal_gov_agent import GovRAG
from nepal_gov_agent import cli as cli_mod
from nepal_gov_agent.cli import main


@pytest.fixture
def use_test_rag(monkeypatch: pytest.MonkeyPatch, rag: GovRAG) -> None:
    monkeypatch.setattr(cli_mod, "_load_rag", lambda corpus: rag)


def test_ask_prints_answer_sources_confidence(use_test_rag, capsys):
    main(["ask", "What does the National AI Centre do?", "--k", "3"])
    out = capsys.readouterr().out
    assert "ANSWER" in out
    assert "SOURCES" in out
    assert "National AI Policy-Final_uxc94vg.pdf" in out
    assert "Confidence:" in out


def test_stats_prints_counts(use_test_rag, capsys):
    main(["stats"])
    out = capsys.readouterr().out
    assert "Documents: 3" in out
    assert "Offline:   True" in out


def test_benchmark_smoke_test(use_test_rag, capsys):
    main(["benchmark"])
    out = capsys.readouterr().out
    assert "Hand-curated smoke test" in out
    assert "Recall@3" in out


def test_benchmark_synthetic(use_test_rag, capsys):
    main(["benchmark", "--synthetic"])
    out = capsys.readouterr().out
    assert "NOT human-validated" in out


@pytest.mark.parametrize("workflow", ["document_qa", "service_guide", "corpus_search"])
def test_agent_workflows(use_test_rag, capsys, monkeypatch, tmp_path, workflow):
    monkeypatch.chdir(tmp_path)  # GovAgent's default session DB is created in the CWD
    main(["agent", "citizenship renewal", "--workflow", workflow, "--session", "cli"])
    out = capsys.readouterr().out
    assert f"Workflow:   {workflow}" in out
    assert "Sources:" in out


def test_missing_corpus_is_a_clean_error(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["stats", "--corpus", str(tmp_path / "nope")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error: Corpus directory not found")
