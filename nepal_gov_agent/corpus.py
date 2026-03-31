"""
Corpus management for Nepal GovAgent.

Design principle:
- The library never writes to your filesystem without explicit action.
- download_corpus() is opt-in — it downloads the seed corpus only when you call it.
- You can always point GovRAG at any folder containing Nepal government PDFs.
"""

from __future__ import annotations

import os
from urllib.parse import quote

import requests

# Seed corpus — same PDFs as github.com/irfanalidv/Nepal-Gov-Agent/tree/main/Data/
# (Filenames must match that folder exactly so raw.githubusercontent.com URLs resolve.)
SEED_CORPUS: list[str] = [
    "1714977234_32.pdf",
    "2082.9.2 प्रतिनिधि सभा सदस्य निर्वाचन (पहिलो संशोधन) अध्यादेश,२०८२_v1cs5ms.pdf",
    "Constitution of Nepal (2nd amd. English)_xf33zb3.pdf",
    "National AI Policy-Final_uxc94vg.pdf",
    "dnf_jbji8eb.pdf",
    "मानव अधिकार पुरस्कार कोष सञ्चालन नियमावली, 2075_n4hme7v.pdf",
]

RAW_BASE_URL = (
    "https://raw.githubusercontent.com/irfanalidv/Nepal-Gov-Agent/main/Data/"
)


def download_corpus(dest_dir: str = "./nepal_gov_data/", force: bool = False) -> str:
    """
    Download the Nepal GovAgent seed corpus to a local folder.

    Pulls the same PDFs as ``Data/`` on GitHub (National AI Policy, Constitution,
    Digital Nepal Framework, election ordinance, human rights fund rules, plus one
    additional indexed government PDF). Nothing is written until you call this.

    Example:
        corpus_dir = download_corpus()
        rag = GovRAG(corpus_dir=corpus_dir)

        # Or use your own PDFs — no download_corpus() call:
        rag = GovRAG(corpus_dir="./my_ministry_docs/")
    """
    abs_dest = os.path.abspath(os.path.expanduser(dest_dir))
    os.makedirs(abs_dest, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "nepal-gov-agent-corpus-download/1.0 "
                "(https://pypi.org/project/nepal-gov-agent/)"
            )
        }
    )

    print(f"📁 Corpus folder: {abs_dest}")
    print(f"📥 Downloading {len(SEED_CORPUS)} Nepal government documents...\n")

    for filename in SEED_CORPUS:
        dest_path = os.path.join(abs_dest, filename)
        if os.path.exists(dest_path) and not force:
            print(f"   ✓ Already exists — skipping: {filename}")
            continue
        url = RAW_BASE_URL + quote(filename, safe="")
        response = session.get(url, timeout=120)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        size_kb = len(response.content) // 1024
        print(f"   ✅ Downloaded: {filename} ({size_kb} KB)")

    print(f"\n✅ Seed corpus ready — {len(SEED_CORPUS)} documents in {abs_dest}")
    print(
        "💡 To use your own PDFs, pass any folder to GovRAG(corpus_dir=...)\n"
    )
    return abs_dest
