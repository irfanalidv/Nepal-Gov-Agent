"""
Synthetic QA pair generation for the Nepal government corpus.

What this is:
    A reproducible harness that uses GPT-4.1-mini to draft QA pairs from each
    page of the seed PDFs, then validates that the expected keywords actually
    appear in the source passage before keeping the pair.

What this is NOT:
    A human-validated benchmark. The pairs are LLM-authored. Use the numbers
    as a smoke test for regressions and a starting point for real evaluation,
    not as ground truth. See ``eval_data/README.md`` for the disclaimer
    distributed with each generated set.

Usage:
    from nepal_gov_agent.eval import generate_synthetic_qa
    pairs = generate_synthetic_qa(
        corpus_dir="Data/",
        out_path="eval_data/synthetic_qa_v1.jsonl",
        model="gpt-4.1-mini",
        max_pairs_per_doc=20,
    )
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import sysconfig
import unicodedata
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..exceptions import CorpusNotFoundError, EmptyCorpusError, MissingDependencyError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SyntheticQAPair:
    """One LLM-generated QA pair plus provenance for honesty."""

    query: str
    expected_keywords: list[str]
    expected_doc: str
    language: str  # "english" | "nepali"
    source_page: int
    source_excerpt: str  # first 240 chars of the page used to generate
    generator_model: str
    is_synthetic: bool = True  # always True for this module; keep explicit


def _to_jsonable(pair: SyntheticQAPair) -> dict[str, Any]:
    return asdict(pair)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You generate evaluation QA pairs from Nepal government policy and legal documents.

Rules:
1. Each question MUST be answerable strictly from the passage provided. No outside knowledge.
2. Generate exactly 2 questions per passage: one in English, one in Nepali (Devanagari).
3. For each question, provide 3-5 "expected_keywords" — short distinctive phrases (1-4 words)
   that MUST appear verbatim in the passage. These will be used to validate retrieval.
   Pick keywords that are specific to the passage, not generic words like "the" or "government".
4. Skip the passage if it is a cover page, table of contents, blank, or has fewer than 40 words
   of meaningful content. Return an empty list in that case.
5. Do not invent facts. If the passage is ambiguous, generate fewer questions or none.

Return STRICT JSON with no preamble or markdown:
{
  "pairs": [
    {"language": "english", "query": "...", "expected_keywords": ["...", "..."]},
    {"language": "nepali",  "query": "...", "expected_keywords": ["...", "..."]}
  ]
}
"""


_USER_TEMPLATE = """Document: {doc_name}
Page: {page}

Passage:
\"\"\"
{passage}
\"\"\"

Generate the QA pairs as instructed."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    """NFC + lowercase + collapse whitespace. Matches preprocess.py spirit."""
    s = unicodedata.normalize("NFC", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def _all_keywords_present(passage: str, keywords: list[str]) -> bool:
    """Every keyword must appear verbatim (case-insensitive, NFC) in the passage."""
    if not keywords:
        return False
    passage_n = _norm(passage)
    for kw in keywords:
        if not kw:
            return False
        if _norm(kw) not in passage_n:
            return False
    return True


def _is_meaningful_passage(text: str, min_words: int = 40) -> bool:
    if not text:
        return False
    words = re.findall(r"\S+", text)
    return len(words) >= min_words


# ---------------------------------------------------------------------------
# OpenAI client (lazy import, optional dep)
# ---------------------------------------------------------------------------


#: Per-request timeout (seconds) for OpenAI calls; the SDK default is 10 minutes.
OPENAI_TIMEOUT = 60.0


def _get_openai_client(api_key: str | None = None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise MissingDependencyError(
            "openai package not installed. Install with: pip install 'nepal-gov-agent[eval]'"
        ) from e
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingDependencyError(
            "OPENAI_API_KEY environment variable not set. "
            "Export your key or pass api_key=... to generate_synthetic_qa."
        )
    return OpenAI(api_key=key, timeout=OPENAI_TIMEOUT, max_retries=2)


def _call_openai(client: Any, model: str, passage: str, doc_name: str, page: int) -> dict[str, Any]:
    """One call, JSON-only response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    doc_name=doc_name, page=page, passage=passage[:3500]
                ),
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Could not parse JSON from model for %s p.%d", doc_name, page)
        return {"pairs": []}
    if not isinstance(data, dict):
        logger.warning("Model returned non-object JSON for %s p.%d", doc_name, page)
        return {"pairs": []}
    return data


# ---------------------------------------------------------------------------
# Passage iteration over the corpus
# ---------------------------------------------------------------------------


def _iter_passages(corpus_dir: str) -> Iterator[tuple[str, int, str]]:
    """
    Yield (doc_name, page_number, page_text) for every page of every PDF
    in corpus_dir. Uses PyMuPDF directly (same library RAGNav's ingest uses)
    to keep this independent of RAGNav's chunking decisions.
    """
    import pymupdf  # hard dependency; imported lazily to keep package import light

    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise CorpusNotFoundError(f"Corpus directory not found: {corpus_dir}")

    pdf_files = sorted(corpus_path.glob("*.pdf"))
    if not pdf_files:
        raise EmptyCorpusError(f"No PDFs found in {corpus_dir}")

    for pdf_path in pdf_files:
        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            logger.warning("Could not open %s: %s", pdf_path.name, e)
            continue
        try:
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text("text") or ""
                except Exception as e:
                    logger.warning("Page %d of %s failed: %s", page_num + 1, pdf_path.name, e)
                    continue
                yield pdf_path.name, page_num + 1, text
        finally:
            # Also runs if the consumer stops early or raises mid-iteration.
            doc.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_synthetic_qa(
    corpus_dir: str = "Data/",
    out_path: str = "eval_data/synthetic_qa_v1.jsonl",
    model: str = "gpt-4.1-mini",
    max_pairs_per_doc: int = 20,
    min_passage_words: int = 40,
    resume: bool = True,
    *,
    api_key: str | None = None,
    client: Any | None = None,
) -> list[SyntheticQAPair]:
    """
    Generate, validate, and persist synthetic QA pairs.

    Args:
        corpus_dir: folder with the seed PDFs.
        out_path:   JSONL file. One pair per line. Idempotent if resume=True.
        model:      OpenAI chat model. Default ``gpt-4.1-mini``.
        max_pairs_per_doc: cap to keep generation bounded and per-doc balanced.
        min_passage_words: skip near-empty pages (covers, dividers).
        resume:     skip pages already present in out_path.
        api_key:    OpenAI key; defaults to the ``OPENAI_API_KEY`` environment variable.
        client:     Pre-built OpenAI-compatible client (anything exposing
                    ``chat.completions.create``). Overrides ``api_key``; handy for
                    tests and for OpenAI-compatible gateways.

    Returns:
        Validated pairs. Pairs whose keywords don't appear in source are dropped.
    """
    if client is None:
        client = _get_openai_client(api_key)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[tuple[str, int]] = set()
    per_doc_counts: dict[str, int] = {}
    kept: list[SyntheticQAPair] = []

    # Resume: load existing pairs so we skip covered pages and return a complete list.
    if resume and out_p.exists():
        for lineno, line in enumerate(out_p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                pair = SyntheticQAPair(**rec)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, out_path, exc)
                continue
            already_done.add((pair.expected_doc, int(pair.source_page)))
            kept.append(pair)
            per_doc_counts[pair.expected_doc] = per_doc_counts.get(pair.expected_doc, 0) + 1
        logger.info("Resume: %d pages already covered in %s", len(already_done), out_path)

    with out_p.open("a", encoding="utf-8") as f:
        for doc_name, page_num, text in _iter_passages(corpus_dir):
            if (doc_name, page_num) in already_done:
                continue
            if per_doc_counts.get(doc_name, 0) >= max_pairs_per_doc:
                continue
            if not _is_meaningful_passage(text, min_words=min_passage_words):
                continue

            logger.info("Generating QA for %s page %d", doc_name, page_num)
            try:
                data = _call_openai(client, model, text, doc_name, page_num)
            except Exception as e:
                logger.warning("OpenAI call failed for %s p.%d: %s", doc_name, page_num, e)
                continue

            for raw in data.get("pairs", []) or []:
                lang = str(raw.get("language", "")).lower().strip()
                q = str(raw.get("query", "")).strip()
                kws = [
                    str(k).strip() for k in (raw.get("expected_keywords") or []) if str(k).strip()
                ]
                if lang not in {"english", "nepali"} or not q or not kws:
                    continue
                if not _all_keywords_present(text, kws):
                    logger.debug(
                        "Dropped pair (kw not in source) %s p.%d lang=%s",
                        doc_name,
                        page_num,
                        lang,
                    )
                    continue

                pair = SyntheticQAPair(
                    query=q,
                    expected_keywords=kws,
                    expected_doc=doc_name,
                    language=lang,
                    source_page=page_num,
                    source_excerpt=text[:240].strip(),
                    generator_model=model,
                )
                kept.append(pair)
                per_doc_counts[doc_name] = per_doc_counts.get(doc_name, 0) + 1
                f.write(json.dumps(_to_jsonable(pair), ensure_ascii=False) + "\n")
                f.flush()

                if per_doc_counts[doc_name] >= max_pairs_per_doc:
                    break

    logger.info(
        "Generation done: %d pairs across %d documents (saved to %s)",
        len(kept),
        len(per_doc_counts),
        out_path,
    )
    return kept


def _resolve_eval_path(path: str) -> Path:
    """Resolve ``path``; for relative paths also look in the install's data dir.

    The wheel ships ``eval_data/`` as setuptools *data-files*, which land under
    ``sys.prefix`` (or the user base for ``pip install --user``), not the CWD.
    Falling back there makes ``benchmark --synthetic`` work right after ``pip install``.
    """
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    bases = [sysconfig.get_path("data"), sys.prefix]
    with contextlib.suppress(KeyError):
        bases.append(sysconfig.get_path("data", f"{os.name}_user"))
    for base in bases:
        if base and (Path(base) / p).exists():
            return Path(base) / p
    return p


def load_synthetic_qa(path: str = "eval_data/synthetic_qa_v1.jsonl") -> list[SyntheticQAPair]:
    """Load a previously-generated synthetic QA set.

    Relative paths are tried against the CWD first, then the package's installed
    data directory (where ``pip install`` places ``eval_data/synthetic_qa_v1.jsonl``).
    """
    p = _resolve_eval_path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Synthetic QA file not found: {path}\n"
            "Run generate_synthetic_qa(...) first, "
            "or `python -m nepal_gov_agent.eval.generate`."
        )
    pairs: list[SyntheticQAPair] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            pairs.append(SyntheticQAPair(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{p}:{lineno}: not a valid SyntheticQAPair record ({exc})") from exc
    return pairs


def to_benchmark_format(pairs: list[SyntheticQAPair]) -> list[dict[str, Any]]:
    """Convert SyntheticQAPair objects to the dict shape ``run_benchmark`` expects."""
    return [
        {
            "query": p.query,
            "expected_keywords": p.expected_keywords,
            "expected_doc": p.expected_doc,
            "language": p.language,
        }
        for p in pairs
    ]
