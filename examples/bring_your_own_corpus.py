"""
Bring your own corpus: custom retrieval settings and per-document filtering
==========================================================================
Shows    : pointing GovRAG at any folder of PDFs, tuning GovRAGConfig (weighted fusion,
           keyword-heavy for legal text), restricting a question to one document with
           ``allowed_doc_ids``, and raw ``search`` for building your own UI.
           Also asks a Nepali question against English PDFs.
Data     : a three-PDF mini corpus generated on the fly (examples/_mini_corpus.py).
           Swap ``corpus_dir`` for your own folder, e.g. "./my_ministry_docs/".
Needs    : ``pip install nepal-gov-agent``. No API key. First run downloads the
           embedding model from Hugging Face.
Run      : python examples/bring_your_own_corpus.py
"""

import logging
import tempfile
from pathlib import Path

from _mini_corpus import build_mini_corpus

from nepal_gov_agent import GovRAG, GovRAGConfig

logging.basicConfig(level=logging.WARNING)

work = Path(tempfile.mkdtemp(prefix="nga_byoc_"))
config = GovRAGConfig(
    fusion="weighted",  # blend scores; the default "rrf" uses ranks only
    w_bm25=0.7,  # lean on exact legal keywords
    w_vec=0.3,
    k_final=3,
    cache_dir=str(work / "cache"),
)
rag = GovRAG(corpus_dir=build_mini_corpus(work / "corpus"), config=config)
print("Index:", rag.stats)

# 1. Restrict a question to one document (doc ids are "pdf:<filename>").
only_licences = {"pdf:driving_licence_rules.pdf"}
r = rag.ask("What do I need to do?", allowed_doc_ids=only_licences)
print("\nScoped to driving_licence_rules.pdf ->", {s["doc"] for s in r.sources})

# 2. Raw ranked blocks, best match first.
for b in rag.search("Transport Management Office trial", k=2):
    print(f"  {b['doc_id']} p.{b['anchors'].get('page')}: {b['content'][:70]}...")

# 3. Nepali question, English documents (dense retrieval does the cross-lingual work;
#    expect lower recall than same-language queries, see the README benchmark).
r = rag.ask("नागरिकता प्रमाणपत्र कहाँ बनाउने?")  # "Where do I get a citizenship certificate?"
print(f"\nNepali query -> top source: {r.sources[0]['doc']} (confidence {r.confidence})")
