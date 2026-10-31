"""
LLM answers with inline citations (stub offline, or a local Ollama model)
=========================================================================
Shows    : ``GovRAG.ask(llm=...)``. Retrieval stays local; the LLM only writes the
           answer and must cite blocks as ``[[block_id]]``; RAGNav rejects answers with an
           uncited sentence or an unknown block id. If the LLM call fails or the
           answer is rejected, GovRAG logs a warning and returns the extractive
           answer instead of raising.
Data     : a three-PDF mini corpus generated on the fly (examples/_mini_corpus.py).
Needs    : Nothing extra by default: a stub LLM that cites the top block runs offline.
           With ``--ollama`` it uses a real local model instead: ``ollama serve`` and
           ``ollama pull qwen2.5:7b`` first (no API key, nothing leaves the machine).
Run      : python examples/cited_answers_llm.py [--ollama]
"""

import logging
import sys
import tempfile
from pathlib import Path

from _mini_corpus import build_mini_corpus
from ragnav.llm.fake import FakeLLMClient

from nepal_gov_agent import GovRAG, GovRAGConfig, OllamaClient

logging.basicConfig(level=logging.WARNING)

work = Path(tempfile.mkdtemp(prefix="nga_llm_"))
rag = GovRAG(
    corpus_dir=build_mini_corpus(work / "corpus"),
    config=GovRAGConfig(cache_dir=str(work / "cache"), k_final=3),
)
question = "Which office issues a replacement citizenship certificate?"


class StubCitingLLM(FakeLLMClient):
    """Pretends to be an LLM: one sentence, citing the best-matching block."""

    def chat(self, *, messages, model=None, temperature=0):
        top = rag.search(question, k=1)[0]
        return (
            "Apply at the District Administration Office with a ward-office "
            f"recommendation [[{top['block_id']}]]."
        )


llm = OllamaClient() if "--ollama" in sys.argv else StubCitingLLM()
result = rag.ask(question, llm=llm, with_citations=True)
print("Answer:", result.answer)
print("Cited block ids:", result.cited_block_ids)
