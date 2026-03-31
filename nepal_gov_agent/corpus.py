"""
Seed corpus download — same PDFs as the repo ``Data/`` folder on GitHub ``main``.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import requests

_RAW_BASE = "https://raw.githubusercontent.com/irfanalidv/Nepal-Gov-Agent/main/Data/"

# Filenames must match ``Data/`` on ``main`` (URL-encoded per segment when fetching).
_SEED_PDFS: tuple[str, ...] = (
    "1714977234_32.pdf",
    "2082.9.2 प्रतिनिधि सभा सदस्य निर्वाचन (पहिलो संशोधन) अध्यादेश,२०८२_v1cs5ms.pdf",
    "Constitution of Nepal (2nd amd. English)_xf33zb3.pdf",
    "National AI Policy-Final_uxc94vg.pdf",
    "dnf_jbji8eb.pdf",
    "मानव अधिकार पुरस्कार कोष सञ्चालन नियमावली, 2075_n4hme7v.pdf",
)


def download_corpus(dest_dir: str = "./nepal_gov_data/", force: bool = False) -> str:
    out = Path(dest_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "nepal-gov-agent-corpus-download/1.0 (https://pypi.org/project/nepal-gov-agent/)"}
    )

    for name in _SEED_PDFS:
        target = out / name
        if target.is_file() and not force:
            print("Skipping (already exists): %s" % name)
            continue

        url = _RAW_BASE + quote(name, safe="")
        response = session.get(url, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                "Failed to download %r (HTTP %s). Check that the file exists on main: %s"
                % (name, response.status_code, url)
            )

        target.write_bytes(response.content)
        print("Downloading: %s ✅" % name)

    return str(out)
