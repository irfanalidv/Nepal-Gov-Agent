"""Error paths, exception hierarchy and edge cases (no network)."""

from __future__ import annotations

from unittest import mock

import pytest
import requests

from nepal_gov_agent import (
    CorpusDownloadError,
    CorpusNotFoundError,
    EmptyCorpusError,
    GovRAG,
    LLMUnavailableError,
    NepalGovAgentError,
    OllamaClient,
    download_corpus,
    run_benchmark,
    run_synthetic_benchmark,
)
from nepal_gov_agent.ingest import ingest_corpus
from nepal_gov_agent.rag import (
    _EMBEDDING_MODEL_MARKER,
    _invalidate_embedding_cache_if_model_changed,
)


@pytest.mark.parametrize(
    "exc, builtin",
    [
        (CorpusNotFoundError, FileNotFoundError),
        (EmptyCorpusError, ValueError),
        (CorpusDownloadError, RuntimeError),
        (LLMUnavailableError, RuntimeError),
    ],
)
def test_exceptions_keep_builtin_bases(exc, builtin):
    assert issubclass(exc, NepalGovAgentError)
    assert issubclass(exc, builtin)


def test_empty_corpus_dir_raises(tmp_path):
    with pytest.raises(EmptyCorpusError, match="No PDF files"):
        ingest_corpus(str(tmp_path))


def test_unparseable_pdfs_raise_instead_of_empty_index(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf at all")
    with pytest.raises(EmptyCorpusError, match="yielded any text"):
        ingest_corpus(str(tmp_path))


def test_govrag_missing_corpus_is_corpus_not_found():
    with pytest.raises(CorpusNotFoundError):
        GovRAG(corpus_dir="definitely_missing_dir_123/")


# --- Ollama ----------------------------------------------------------------


def _post_raising(exc: Exception):
    return mock.patch("nepal_gov_agent.llm_ollama.requests.post", side_effect=exc)


def test_ollama_timeout_is_reported():
    with (
        _post_raising(requests.exceptions.Timeout()),
        pytest.raises(LLMUnavailableError, match="did not answer within 5s"),
    ):
        OllamaClient(timeout=5).chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_http_error_names_the_model():
    resp = mock.Mock(status_code=404)
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    with (
        mock.patch("nepal_gov_agent.llm_ollama.requests.post", return_value=resp),
        pytest.raises(LLMUnavailableError, match=r"HTTP 404 for model 'tiny'"),
    ):
        OllamaClient(model="tiny").chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_malformed_reply():
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"unexpected": True}
    with (
        mock.patch("nepal_gov_agent.llm_ollama.requests.post", return_value=resp),
        pytest.raises(LLMUnavailableError, match="Unexpected reply"),
    ):
        OllamaClient().chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_passes_timeout_through():
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": "ok"}}
    with mock.patch("nepal_gov_agent.llm_ollama.requests.post", return_value=resp) as post:
        OllamaClient(timeout=7).chat(messages=[])
    assert post.call_args.kwargs["timeout"] == 7


# --- download ---------------------------------------------------------------


def test_download_network_error_is_wrapped(tmp_path):
    with (
        mock.patch(
            "requests.Session.get", side_effect=requests.exceptions.ConnectionError("offline")
        ),
        pytest.raises(CorpusDownloadError, match="Could not download"),
    ):
        download_corpus(str(tmp_path))
    assert not list(tmp_path.iterdir())  # nothing half-written


def test_download_http_error_is_wrapped(tmp_path):
    resp = mock.Mock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
    with mock.patch("requests.Session.get", return_value=resp), pytest.raises(CorpusDownloadError):
        download_corpus(str(tmp_path))


# --- benchmark --------------------------------------------------------------


def test_benchmark_custom_pairs(rag):
    pairs = [
        {
            "query": "National AI Centre secretariat",
            "expected_keywords": ["secretariat"],
            "expected_doc": "National AI Policy-Final_uxc94vg.pdf",
            "language": "english",
        },
        {
            "query": "zzz unrelated",
            "expected_keywords": [],
            "expected_doc": "",
            "language": "nepali",
        },
    ]
    res = run_benchmark(rag, qa_pairs=pairs, verbose=True, eval_kind="custom")
    assert res.total_queries == 2
    assert res.keyword_hit_rate == pytest.approx(0.5)
    assert res.recall_at_1 == pytest.approx(0.5)  # best match is ranked first
    assert res.english_recall == pytest.approx(1.0)
    assert res.nepali_recall == 0.0
    assert "Custom eval set" in res.report()
    assert res.per_query[0]["doc_hit"] is True


def test_benchmark_empty_pairs_raises(rag):
    with pytest.raises(ValueError, match="qa_pairs is empty"):
        run_benchmark(rag, qa_pairs=[])


def test_benchmark_rejects_bad_k(rag):
    with pytest.raises(ValueError, match="k_values"):
        run_benchmark(rag, k_values=[0, 3])


def test_synthetic_benchmark_empty_file(rag, tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No pairs loaded"):
        run_synthetic_benchmark(rag, path=str(path))


# --- embedding-cache invalidation ------------------------------------------


def test_cache_kept_when_model_unchanged(tmp_path):
    (tmp_path / "embeddings.db").write_bytes(b"x")
    (tmp_path / _EMBEDDING_MODEL_MARKER).write_text("model-a", encoding="utf-8")
    _invalidate_embedding_cache_if_model_changed(tmp_path, "model-a")
    assert (tmp_path / "embeddings.db").exists()


def test_cache_cleared_when_model_changes(tmp_path):
    (tmp_path / "embeddings.db").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / _EMBEDDING_MODEL_MARKER).write_text("model-a", encoding="utf-8")
    _invalidate_embedding_cache_if_model_changed(tmp_path, "model-b")
    assert list(tmp_path.iterdir()) == []


def test_legacy_cache_without_marker_is_rebuilt(tmp_path):
    (tmp_path / "embeddings.db").write_bytes(b"x")
    _invalidate_embedding_cache_if_model_changed(tmp_path, "model-a")
    assert not (tmp_path / "embeddings.db").exists()


def test_missing_or_fresh_cache_dir_is_noop(tmp_path):
    _invalidate_embedding_cache_if_model_changed(tmp_path / "missing", "m")
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    _invalidate_embedding_cache_if_model_changed(tmp_path, "m")
    assert (tmp_path / "notes.txt").exists()


# --- agent ------------------------------------------------------------------


def test_unknown_workflow_does_not_touch_memory(agent):
    agent.clear_history()
    with pytest.raises(ValueError, match="Unknown workflow"):
        agent.run("query", workflow="nope")
    assert agent.history == []


def test_report_marks_missing_language_as_na(rag):
    pairs = [{"query": "AI policy", "expected_keywords": ["ai"], "language": "english"}]
    report = run_benchmark(rag, qa_pairs=pairs, verbose=False).report()
    assert "Nepali recall@3:    n/a" in report
    assert "English recall@3:   1.000" in report
