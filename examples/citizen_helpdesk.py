"""
Citizen helpdesk: eligibility + procedure answers with session memory
====================================================================
Shows    : GovAgent's ``service_guide`` workflow (two retrieval passes: "who is
           eligible" then "what are the steps"), plus ``document_qa`` follow-ups
           stored in a per-citizen SQLite session.
Data     : a three-PDF mini corpus generated on the fly (examples/_mini_corpus.py).
Needs    : ``pip install nepal-gov-agent``. No API key, no LLM. The first run
           downloads the embedding model (~470 MB) from Hugging Face; later runs
           are fully offline.
Run      : python examples/citizen_helpdesk.py
"""

import logging
import tempfile
from pathlib import Path

from _mini_corpus import build_mini_corpus

from nepal_gov_agent import GovAgent, GovAgentConfig, GovRAG, GovRAGConfig

logging.basicConfig(level=logging.WARNING)

work = Path(tempfile.mkdtemp(prefix="nga_helpdesk_"))
rag = GovRAG(
    corpus_dir=build_mini_corpus(work / "corpus"),
    config=GovRAGConfig(cache_dir=str(work / "cache")),
)
agent = GovAgent(
    rag=rag,
    config=GovAgentConfig(session_db=str(work / "sessions.db")),
    session_id="citizen-42",
)

guide = agent.run("replacement citizenship certificate", workflow="service_guide", k_final=2)
print(guide.answer)
print(f"\nconfidence={guide.confidence}  steps={len(guide.steps)}")
print("cited:", sorted({(s["doc"], s["page"]) for s in guide.sources}))

follow_up = agent.run("How old must I be for a driving licence?", k_final=1)
print("\nFollow-up:", follow_up.answer.splitlines()[1])
print(f"Session {follow_up.session_id!r} now holds {len(agent.history)} messages.")
