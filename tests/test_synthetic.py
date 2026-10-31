"""Synthetic QA generation with a fake OpenAI client (no network, no key)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from nepal_gov_agent import MissingDependencyError, generate_synthetic_qa, load_synthetic_qa
from nepal_gov_agent.eval import generate as generate_cli
from nepal_gov_agent.eval.synthetic import _call_openai, _get_openai_client, _iter_passages


class FakeCompletions:
    """Returns one English pair whose keyword is in the passage and one that is not."""

    def __init__(self, content: str | None = None) -> None:
        self.calls = 0
        self._content = content

    def create(self, **kwargs):
        self.calls += 1
        if self._content is not None:
            content = self._content
        else:
            passage = kwargs["messages"][1]["content"]
            word = "AI" if "AI" in passage else passage.split('"""')[1].split()[0]
            content = json.dumps(
                {
                    "pairs": [
                        {"language": "english", "query": "Q?", "expected_keywords": [word]},
                        {"language": "nepali", "query": "प्रश्न?", "expected_keywords": ["absent"]},
                        {"language": "klingon", "query": "x", "expected_keywords": [word]},
                    ]
                }
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_client(content: str | None = None):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(content)))


def test_generate_validates_and_persists(corpus_dir, tmp_path):
    out = tmp_path / "qa.jsonl"
    client = fake_client()
    pairs = generate_synthetic_qa(
        corpus_dir=corpus_dir, out_path=str(out), min_passage_words=5, client=client
    )
    # 6 pages in the synthetic corpus; each yields exactly one valid pair.
    assert len(pairs) == 6
    assert all(p.language == "english" and p.is_synthetic for p in pairs)
    assert load_synthetic_qa(str(out)) == pairs

    # Resume: nothing new to generate, no further API calls, same pairs returned.
    calls = client.chat.completions.calls
    again = generate_synthetic_qa(
        corpus_dir=corpus_dir, out_path=str(out), min_passage_words=5, client=client
    )
    assert again == pairs
    assert client.chat.completions.calls == calls


def test_generate_respects_per_doc_cap(corpus_dir, tmp_path):
    pairs = generate_synthetic_qa(
        corpus_dir=corpus_dir,
        out_path=str(tmp_path / "qa.jsonl"),
        min_passage_words=5,
        max_pairs_per_doc=1,
        client=fake_client(),
    )
    docs = [p.expected_doc for p in pairs]
    assert len(docs) == len(set(docs)) == 3


def test_resume_skips_malformed_lines(corpus_dir, tmp_path, caplog):
    out = tmp_path / "qa.jsonl"
    out.write_text("{not json}\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        pairs = generate_synthetic_qa(
            corpus_dir=corpus_dir, out_path=str(out), min_passage_words=5, client=fake_client()
        )
    assert "Skipping malformed line 1" in caplog.text
    assert len(pairs) == 6


def test_short_pages_are_skipped(corpus_dir, tmp_path):
    client = fake_client()
    pairs = generate_synthetic_qa(
        corpus_dir=corpus_dir,
        out_path=str(tmp_path / "qa.jsonl"),
        min_passage_words=10_000,
        client=client,
    )
    assert pairs == []
    assert client.chat.completions.calls == 0


def test_bad_model_json_is_tolerated():
    assert _call_openai(fake_client("not json"), "m", "p", "d", 1) == {"pairs": []}
    assert _call_openai(fake_client("[1, 2]"), "m", "p", "d", 1) == {"pairs": []}


def test_iter_passages_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(_iter_passages(str(tmp_path / "missing")))
    with pytest.raises(ValueError):
        list(_iter_passages(str(tmp_path)))


def test_iter_passages_skips_unopenable_pdf(tmp_path, corpus_dir):
    (tmp_path / "broken.pdf").write_bytes(b"garbage")
    assert list(_iter_passages(str(tmp_path))) == []


def test_missing_key_is_explained(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingDependencyError, match="OPENAI_API_KEY"):
        _get_openai_client()


def test_missing_openai_package_is_explained(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "openai", None)
    with pytest.raises(MissingDependencyError, match=r"nepal-gov-agent\[eval\]"):
        _get_openai_client("dummy-key")


def test_load_synthetic_qa_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"query": "only"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:1"):
        load_synthetic_qa(str(path))


def test_generate_cli_summary(capsys, tmp_path):
    from nepal_gov_agent import SyntheticQAPair

    pair = SyntheticQAPair("Q?", ["k"], "doc.pdf", "english", 1, "x", "m")
    with mock.patch.object(generate_cli, "generate_synthetic_qa", return_value=[pair]) as gen:
        generate_cli.main(["--corpus", "c/", "--out", str(tmp_path / "o.jsonl"), "--no-resume"])
    assert gen.call_args.kwargs["resume"] is False
    out = capsys.readouterr().out
    assert "Total pairs:      1" in out
    assert "doc.pdf" in out
