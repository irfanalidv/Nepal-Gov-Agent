"""Build a tiny, self-contained PDF corpus for the offline examples.

The passages are short paraphrases written for these demos. They mimic the
style of Nepal policy documents but are NOT official text; use
``nepal_gov_agent.download_corpus()`` for the real seed PDFs.

All text is English: PyMuPDF's built-in fonts cannot write extractable Devanagari.
Nepali questions still work because the default embedding model
(``intfloat/multilingual-e5-small``) maps both languages into one vector space.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pymupdf

MINI_DOCS: dict[str, list[str]] = {
    "citizenship_services_guide.pdf": [
        "Citizenship certificate renewal. Eligibility: any Nepali citizen whose certificate "
        "is damaged, lost or carries outdated details may apply for a replacement copy.",
        "Procedure for a replacement citizenship certificate. Submit the application form at "
        "the District Administration Office with a recommendation from the ward office, two "
        "passport-size photographs and a copy of the old certificate if available.",
    ],
    "driving_licence_rules.pdf": [
        "Driving licence eligibility. Applicants must be at least 18 years old for a car or "
        "motorcycle licence and must pass a medical fitness check.",
        "Driving licence procedure. Register online, pass the written test, then pass the "
        "practical trial at the Transport Management Office and pay the licence fee.",
    ],
    "national_ai_policy_summary.pdf": [
        "National AI Policy summary. Vision: an ethical, human-centric, inclusive and "
        "sustainable use of artificial intelligence for national development.",
        "The National AI Centre acts as the secretariat for coordination, research and "
        "regulation of artificial intelligence.",
    ],
}


def build_mini_corpus(dest: str | Path | None = None) -> str:
    """Write one PDF per entry in ``MINI_DOCS`` and return the folder path."""
    folder = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="nga_mini_corpus_"))
    folder.mkdir(parents=True, exist_ok=True)
    for name, pages in MINI_DOCS.items():
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
        doc.save(str(folder / name))
        doc.close()
    return str(folder)
