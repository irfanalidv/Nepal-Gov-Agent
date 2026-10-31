"""Fast unit tests that need no index or network (HTTP calls are mocked)."""

import json
from importlib.metadata import version
from unittest import mock

import pytest
import requests

import nepal_gov_agent
from nepal_gov_agent import BenchmarkResult, OllamaClient, load_synthetic_qa
from nepal_gov_agent.cli import main
from nepal_gov_agent.config import DEFAULT_RAG_CONFIG, GovRAGConfig
from nepal_gov_agent.corpus import RAW_BASE_URL, SEED_CORPUS, download_corpus
from nepal_gov_agent.eval.synthetic import (
    SyntheticQAPair,
    _all_keywords_present,
    to_benchmark_format,
)
from nepal_gov_agent.preprocess import preprocess_query


def test_version_matches_metadata():
    assert nepal_gov_agent.__version__ == version("nepal-gov-agent")


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert nepal_gov_agent.__version__ in capsys.readouterr().out


def test_cli_no_command_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1


def test_default_config_is_balanced_and_frozen():
    assert DEFAULT_RAG_CONFIG.w_bm25 + DEFAULT_RAG_CONFIG.w_vec == pytest.approx(1.0)
    with pytest.raises(AttributeError):
        DEFAULT_RAG_CONFIG.k_final = 3  # type: ignore[misc]
    assert GovRAGConfig(k_final=3).k_final == 3


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        ("  नागरिकता कसरी?  ", "नागरिकता"),
        ("Digital Nepal\tFramework", "Digital Nepal Framework"),
    ],
)
def test_preprocess_query_cases(raw, expected):
    assert preprocess_query(raw) == expected


def test_ollama_chat_posts_and_parses():
    resp = mock.Mock()
    resp.json.return_value = {"message": {"content": "namaste"}}
    resp.raise_for_status.return_value = None
    with mock.patch("nepal_gov_agent.llm_ollama.requests.post", return_value=resp) as post:
        out = OllamaClient(base_url="http://host:1/").chat(
            messages=[{"role": "user", "content": "hi"}]
        )
    assert out == "namaste"
    url = post.call_args.args[0]
    assert url == "http://host:1/api/chat"
    assert post.call_args.kwargs["json"]["model"] == "qwen2.5:7b"


def test_ollama_unreachable_raises_runtime_error():
    with mock.patch(
        "nepal_gov_agent.llm_ollama.requests.post",
        side_effect=requests.exceptions.ConnectionError("down"),
    ):
        with pytest.raises(RuntimeError, match="Ollama is not reachable"):
            OllamaClient().chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_embed_not_supported():
    with pytest.raises(RuntimeError):
        OllamaClient().embed(inputs=["x"])


def test_download_corpus_mocked(tmp_path):
    resp = mock.Mock(content=b"%PDF-1.4 fake")
    resp.raise_for_status.return_value = None
    with mock.patch("requests.Session.get", return_value=resp) as get:
        out = download_corpus(str(tmp_path))
    assert get.call_count == len(SEED_CORPUS)
    assert all(c.args[0].startswith(RAW_BASE_URL) for c in get.call_args_list)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(SEED_CORPUS)
    assert out == str(tmp_path)

    # Second call skips existing files unless force=True.
    with mock.patch("requests.Session.get", return_value=resp) as get:
        download_corpus(str(tmp_path))
    assert get.call_count == 0


def test_keyword_validation_is_normalized():
    assert _all_keywords_present("The National AI Centre", ["national ai", "centre"])
    assert not _all_keywords_present("The National AI Centre", ["missing"])


def test_load_synthetic_qa_roundtrip(tmp_path):
    pair = SyntheticQAPair(
        query="What is X?",
        expected_keywords=["x"],
        expected_doc="doc.pdf",
        language="english",
        source_page=1,
        source_excerpt="X is ...",
        generator_model="test-model",
    )
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps(pair.__dict__) + "\n\n", encoding="utf-8")
    loaded = load_synthetic_qa(str(path))
    assert loaded == [pair]
    assert to_benchmark_format(loaded)[0]["expected_doc"] == "doc.pdf"


def test_load_synthetic_qa_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_synthetic_qa(str(tmp_path / "nope.jsonl"))


def test_shipped_synthetic_set_loads():
    pairs = load_synthetic_qa()
    assert len(pairs) > 10
    assert all(p.is_synthetic for p in pairs)
    assert {p.language for p in pairs} <= {"english", "nepali"}


def test_synthetic_report_is_labeled():
    res = BenchmarkResult(
        total_queries=1,
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        keyword_hit_rate=1.0,
        doc_hit_rate=1.0,
        nepali_recall=0.0,
        english_recall=1.0,
        per_query=[],
        eval_kind="synthetic",
        is_synthetic=True,
        generator_model="gpt-x",
    )
    report = res.report()
    assert "NOT human-validated" in report
    assert "gpt-x" in report


def test_cli_download_subcommand(tmp_path):
    with mock.patch("nepal_gov_agent.corpus.download_corpus") as dl:
        main(["download", "--dest", str(tmp_path), "--force"])
    dl.assert_called_once_with(dest_dir=str(tmp_path), force=True)
