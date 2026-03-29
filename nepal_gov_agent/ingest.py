"""
Corpus ingestion pipeline for Nepal government documents.

Handles:
- PDFs (Nepali Devanagari + English)
- Legal document structure (numbered sections, subsections)
- Mixed language detection
- Batch ingestion from a directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ragnav.ingest.pdf import PdfIngestOptions, ingest_pdf_file
from ragnav.models import Block, Document


def ingest_corpus(
    corpus_dir: str,
    *,
    max_pages_per_doc: Optional[int] = None,
    verbose: bool = True,
) -> tuple[list[Document], list[Block]]:
    """
    Ingest all PDFs from a directory into RAGNav documents and blocks.

    Uses page-level PDF ingestion (PyMuPDF). For paper-style structure and
    cross-references, use RAGNav's paper/graph ingest paths separately.

    Args:
        corpus_dir: Path to folder containing PDF files (e.g. "Data/")
        max_pages_per_doc: Limit pages per document (None = all pages)
        verbose: Print ingestion progress

    Returns:
        Tuple of (documents, blocks) ready for RAGNavIndex.build()
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    pdf_files = list(corpus_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {corpus_dir}")

    all_documents: list[Document] = []
    all_blocks: list[Block] = []
    failed = 0

    for pdf_path in pdf_files:
        if verbose:
            print(f"  Ingesting: {pdf_path.name}")

        opts = PdfIngestOptions(
            max_pages=max_pages_per_doc,
            paper_mode=False,  # gov docs are not papers
        )

        try:
            doc, blocks = ingest_pdf_file(
                str(pdf_path),
                name=pdf_path.name,
                metadata={
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "corpus": "nepal_gov",
                },
                opts=opts,
            )
            all_documents.append(doc)
            all_blocks.extend(blocks)

            if verbose:
                print(f"    → {len(blocks)} blocks extracted")

        except Exception as e:
            failed += 1
            if verbose:
                print(f"    ✗ Failed: {e}")
            continue

    if verbose:
        if failed:
            print(f"  Warning: {failed} document(s) failed to ingest")
        print(f"\nCorpus ready: {len(all_documents)} documents, {len(all_blocks)} blocks total")

    return all_documents, all_blocks
