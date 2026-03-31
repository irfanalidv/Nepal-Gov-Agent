"""
Query normalization for retrieval against Nepal government PDFs (Nepali + English).
"""

from __future__ import annotations

import re
import unicodedata

NEPALI_QUESTION_SUFFIXES: tuple[str, ...] = (
    "के हो?",
    "के हो",
    "कसरी?",
    "कसरी",
    "के छ?",
    "के छ",
    "कहाँ?",
    "कहाँ",
    "किन?",
    "किन",
    "कति?",
    "कति",
)


def preprocess_query(query: str) -> str:
    """
    Normalize query for better retrieval against Nepal government PDFs.

    1. Unicode NFC normalization — reduces Devanagari mismatches vs PDF text.
    2. Strip trailing Nepali question markers (helps BM25 keyword overlap).
    3. Collapse whitespace.
    """
    if not query:
        return query
    q = unicodedata.normalize("NFC", query.strip())
    for suffix in NEPALI_QUESTION_SUFFIXES:
        if q.endswith(suffix):
            q = q[: -len(suffix)].strip()
            break
    q = re.sub(r"\s+", " ", q).strip()
    return q
