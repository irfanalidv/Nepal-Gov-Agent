"""
Corpus ingestion pipeline for Nepal government documents.

Handles:
- PDFs (Nepali Devanagari + English)
- Legal document structure (numbered sections, subsections)
- Mixed language detection
- Batch ingestion from a directory
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ragnav.ingest.pdf import PdfIngestOptions, ingest_pdf_file
from ragnav.models import Block, Document

logger = logging.getLogger(__name__)


def ingest_corpus(
    corpus_dir: str,
    *,
    max_pages_per_doc: Optional[int] = None,
) -> tuple[list[Document], list[Block]]:
    """
    Page-level PDF ingestion via PyMuPDF; legal/graph paths live in RAGNav separately.

    Scans only ``corpus_dir`` itself (no subdirectories); use a flat folder e.g. ``Data/*.pdf``.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}\n"
            "Create the directory, add PDF files, and pass the correct path."
        )

    pdf_files = list(corpus_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {corpus_dir}\n"
            "Add .pdf files to that folder (not subfolders) and retry."
        )

    all_documents: list[Document] = []
    all_blocks: list[Block] = []
    failed = 0

    for pdf_path in pdf_files:
        logger.info("Ingesting: %s", pdf_path.name)

        opts = PdfIngestOptions(
            max_pages=max_pages_per_doc,
            paper_mode=False,
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

            logger.info("Extracted %d blocks from %s", len(blocks), pdf_path.name)

        except Exception as e:
            failed += 1
            logger.warning("Failed to ingest %s: %s", pdf_path.name, e)
            continue

    if failed:
        logger.warning("%d document(s) failed to ingest", failed)
    logger.info(
        "Corpus ready: %d documents, %d blocks total",
        len(all_documents),
        len(all_blocks),
    )

    return all_documents, all_blocks
