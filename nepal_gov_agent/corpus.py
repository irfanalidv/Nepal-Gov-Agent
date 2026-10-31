"""
Corpus management for Nepal GovAgent.

Design principle:
- The library never writes to your filesystem without explicit action.
- download_corpus() is opt-in — it downloads the seed corpus only when you call it.
- You can always point GovRAG at any folder containing Nepal government PDFs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import requests

from .exceptions import CorpusDownloadError

logger = logging.getLogger(__name__)

# Seed corpus — five policy/gazette PDFs (not stored on main; pinned release for download).
# Omits 1714977234_32.pdf (Nepal Law Commission *Legal Maxims* — ~500+ blocks) so it does not
# drown smaller Nepali ordinance PDFs in hybrid retrieval for demos and Colab.
SEED_CORPUS: list[str] = [
    "2082.9.2 प्रतिनिधि सभा सदस्य निर्वाचन (पहिलो संशोधन) अध्यादेश,२०८२_v1cs5ms.pdf",
    "Constitution of Nepal (2nd amd. English)_xf33zb3.pdf",
    "National AI Policy-Final_uxc94vg.pdf",
    "dnf_jbji8eb.pdf",
    "मानव अधिकार पुरस्कार कोष सञ्चालन नियमावली, 2075_n4hme7v.pdf",
]

# Pinned tag — PDFs removed from main branch; seed files remain on this release tree.
RAW_BASE_URL = "https://raw.githubusercontent.com/irfanalidv/Nepal-Gov-Agent/v0.4.0/Data/"

#: Seconds to wait for each PDF (connect + read).
DOWNLOAD_TIMEOUT = 120


def download_corpus(dest_dir: str = "./nepal_gov_data/", force: bool = False) -> str:
    """
    Download the Nepal GovAgent seed corpus to a local folder.

    Pulls five policy/gazette PDFs from the pinned release tree (National AI Policy,
    Constitution, Digital Nepal Framework, election ordinance, human rights fund
    rules). Nothing is written until you call this. Each file is written to a
    temporary ``*.part`` name first and renamed once complete, so an interrupted
    download never leaves a truncated PDF that later runs would silently skip.

    Progress is reported through the ``nepal_gov_agent.corpus`` logger.

    Args:
        dest_dir: Folder to write the PDFs into (created if missing).
        force: Re-download files that already exist.

    Returns:
        The absolute path of ``dest_dir``.

    Raises:
        CorpusDownloadError: A file could not be fetched (network error or HTTP status).

    Example:
        corpus_dir = download_corpus()
        rag = GovRAG(corpus_dir=corpus_dir)

        # Or use your own PDFs — no download_corpus() call:
        rag = GovRAG(corpus_dir="./my_ministry_docs/")
    """
    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Corpus folder: %s", dest)
    logger.info("Downloading %d Nepal government documents", len(SEED_CORPUS))

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "nepal-gov-agent-corpus-download/1.0 (https://pypi.org/project/nepal-gov-agent/)"
                )
            }
        )
        for filename in SEED_CORPUS:
            dest_path = dest / filename
            if dest_path.exists() and not force:
                logger.info("Already exists, skipping: %s", filename)
                continue
            url = RAW_BASE_URL + quote(filename, safe="")
            try:
                response = session.get(url, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise CorpusDownloadError(
                    f"Could not download {filename} from {url}: {exc}"
                ) from exc
            tmp_path = dest_path.with_name(dest_path.name + ".part")
            tmp_path.write_bytes(response.content)
            tmp_path.replace(dest_path)
            logger.info("Downloaded: %s (%d KB)", filename, len(response.content) // 1024)

    logger.info("Seed corpus ready: %d documents in %s", len(SEED_CORPUS), dest)
    return str(dest)
