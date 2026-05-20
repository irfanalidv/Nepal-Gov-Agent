"""
Hosted Gradio demo for Nepal GovAgent.

Runs at: https://nepalgov.datacortex.in

Loads the seed corpus once at startup (cached embeddings), exposes three tabs:
  1. Ask         — single-shot QA with confidence + cited sources
  2. Service Guide — pipeline workflow (eligibility → procedure)
  3. Raw Search  — top-k retrieval blocks without synthesis

No OpenAI / no Ollama by default. Pure offline answer assembly from RAGNav
inline-citation logic + FakeLLMClient. If OPENAI_API_KEY or OLLAMA_HOST is
set in the environment, the demo could be upgraded later — out of scope for v0.4.

Usage (local):
    pip install -e ".[demo]"
    python demo/app.py
    # → http://127.0.0.1:7860

Usage (VPS, behind Caddy):
    systemd service runs `python demo/app.py --host 127.0.0.1 --port 7860`
    Caddy reverse-proxies nepalgov.datacortex.in → :7860
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from typing import Any

import gradio as gr

from nepal_gov_agent import GovAgent, GovRAG, GovRAGConfig, download_corpus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

def _boot() -> tuple[GovRAG, GovAgent]:
    """Initialize RAG + agent once. Embeddings cached in CACHE_DIR."""
    corpus_dir = os.environ.get("NGA_CORPUS_DIR", "").strip()
    if not corpus_dir:
        corpus_dir = download_corpus(dest_dir="./nepal_gov_data/")

    cache_dir = os.environ.get("NGA_CACHE_DIR", ".nepal_gov_cache")
    config = GovRAGConfig(cache_dir=cache_dir)
    rag = GovRAG(corpus_dir=corpus_dir, config=config)
    agent = GovAgent(rag=rag, session_id="demo")
    logger.info("Boot complete: %s", rag.stats)
    return rag, agent


# Gradio's queue runs handlers on worker threads; RAGNav's SQLite cache is
# bound to the thread that created it. Lazy-init per thread avoids
# "SQLite objects created in a thread can only be used in that same thread".
_worker = threading.local()


def _get_rag_agent() -> tuple[GovRAG, GovAgent]:
    rag = getattr(_worker, "rag", None)
    if rag is None:
        rag, agent = _boot()
        _worker.rag = rag
        _worker.agent = agent
    return _worker.rag, _worker.agent


# ---------------------------------------------------------------------------
# UI handlers
# ---------------------------------------------------------------------------

_CONFIDENCE_BADGE = {
    "high": "🟢 high",
    "medium": "🟡 medium",
    "low": "🔴 low",
}


def _format_sources_md(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines: list[str] = []
    for i, src in enumerate(sources[:8], 1):
        doc = src.get("doc", "?")
        page = src.get("page", "?")
        heading = src.get("heading") or ""
        excerpt = (src.get("excerpt") or "").strip()
        head = f"**{i}. {doc}** — page {page}"
        if heading:
            head += f"  \n*Section: {heading}*"
        if excerpt:
            head += f"\n\n> {excerpt}"
        lines.append(head)
    return "\n\n---\n\n".join(lines)


def handle_ask(query: str) -> tuple[str, str, str]:
    query = (query or "").strip()
    if not query:
        return "Please enter a question.", "", "—"
    rag, _ = _get_rag_agent()
    try:
        result = rag.ask(query)
    except Exception as e:
        logger.exception("ask failed")
        return f"Error: {e}", "", "—"
    badge = _CONFIDENCE_BADGE.get(result.confidence, result.confidence)
    sources_md = _format_sources_md(result.sources)
    return result.answer, sources_md, badge


def handle_service_guide(query: str) -> tuple[str, str, str]:
    query = (query or "").strip()
    if not query:
        return "Please enter a service question.", "", "—"
    _, agent = _get_rag_agent()
    try:
        result = agent.run(query, workflow="service_guide")
    except Exception as e:
        logger.exception("service_guide failed")
        return f"Error: {e}", "", "—"
    badge = _CONFIDENCE_BADGE.get(result.confidence, result.confidence)
    return result.answer, _format_sources_md(result.sources), badge


def handle_search(query: str, k: int) -> str:
    query = (query or "").strip()
    if not query:
        return "Please enter a search query."
    rag, _ = _get_rag_agent()
    try:
        blocks = rag.search(query, k=k)
    except Exception as e:
        logger.exception("search failed")
        return f"Error: {e}"
    if not blocks:
        return "_No matches._"
    out: list[str] = []
    for i, b in enumerate(blocks, 1):
        doc = (b.get("doc_id") or "").replace("pdf:", "")
        anchors = b.get("anchors") or {}
        page = anchors.get("page", "?")
        content = (b.get("content") or "").strip()
        snippet = content[:600] + ("…" if len(content) > 600 else "")
        out.append(f"**{i}. {doc}** — page {page}\n\n{snippet}")
    return "\n\n---\n\n".join(out)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_INTRO = """\
# Nepal GovAgent — Live Demo

Reference RAG implementation over Nepal's policy and legal corpus.
Hybrid retrieval (BM25 + multilingual embeddings), inline citations, fully
offline-capable. Built on
[RAGNav](https://pypi.org/project/ragnav/),
[ragfallback](https://pypi.org/project/ragfallback/),
and [AgentEnsemble](https://pypi.org/project/agentensemble/).

**Status: maintenance.** This is a published reference impl, not an actively
roadmapped product. PRs welcome. Code:
[github.com/irfanalidv/Nepal-Gov-Agent](https://github.com/irfanalidv/Nepal-Gov-Agent).

Corpus loaded: National AI Policy, Constitution (2nd amd. English),
Digital Nepal Framework, election ordinance, human rights fund rules.

⚠️ The answer is assembled from retrieved passages without an LLM
synthesis layer in this demo — what you see is exactly what the retriever
returns, verbatim from the source PDFs. Confidence and citations are real.
"""

_EXAMPLES_ASK = [
    "What is the vision of Nepal's National AI Policy?",
    "What are the four pillars of Nepal's AI readiness?",
    "How many AI professionals does Nepal aim to train?",
    "What is Digital Nepal Framework 2.0?",
    "नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?",
]

_EXAMPLES_SERVICE = [
    "How do I get a citizenship certificate in Nepal?",
    "What are the eligibility rules for the Human Rights Award Fund?",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Nepal GovAgent — Demo",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    ) as ui:
        gr.Markdown(_INTRO)

        with gr.Tabs():
            # ---------------- Tab 1: Ask
            with gr.Tab("Ask"):
                with gr.Row():
                    q1 = gr.Textbox(
                        label="Question (Nepali or English)",
                        placeholder="e.g. What is Nepal's National AI Policy vision?",
                        lines=2,
                    )
                btn1 = gr.Button("Ask", variant="primary")
                with gr.Row():
                    badge1 = gr.Textbox(label="Retrieval confidence", interactive=False)
                a1 = gr.Markdown(label="Answer")
                s1 = gr.Markdown(label="Sources")
                gr.Examples(_EXAMPLES_ASK, inputs=q1)
                btn1.click(fn=handle_ask, inputs=[q1], outputs=[a1, s1, badge1])

            # ---------------- Tab 2: Service Guide
            with gr.Tab("Service Guide"):
                gr.Markdown(
                    "Pipeline workflow: eligibility → procedure. "
                    "Best for *'how do I do X'* questions."
                )
                q2 = gr.Textbox(
                    label="Service question",
                    placeholder="e.g. How do I apply for the Human Rights Award?",
                    lines=2,
                )
                btn2 = gr.Button("Run pipeline", variant="primary")
                with gr.Row():
                    badge2 = gr.Textbox(label="Confidence", interactive=False)
                a2 = gr.Markdown(label="Answer")
                s2 = gr.Markdown(label="Sources")
                gr.Examples(_EXAMPLES_SERVICE, inputs=q2)
                btn2.click(
                    fn=handle_service_guide,
                    inputs=[q2],
                    outputs=[a2, s2, badge2],
                )

            # ---------------- Tab 3: Raw Search
            with gr.Tab("Raw Search"):
                gr.Markdown(
                    "Hybrid BM25 + multilingual-e5 retrieval. "
                    "No synthesis — top-k blocks straight from the index."
                )
                q3 = gr.Textbox(
                    label="Search query",
                    placeholder="e.g. fundamental rights",
                    lines=2,
                )
                k3 = gr.Slider(1, 20, value=5, step=1, label="Top-k blocks")
                btn3 = gr.Button("Search", variant="primary")
                a3 = gr.Markdown(label="Blocks")
                btn3.click(
                    fn=lambda q, k: handle_search(q, int(k)),
                    inputs=[q3, k3],
                    outputs=[a3],
                )

        gr.Markdown(
            "---\n_Index builds on first question (~30s, cached after that). "
            "Six seed PDFs · hybrid retrieval · offline by default._"
        )
    return ui


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public gradio.live tunnel (local dev only)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    ui = build_ui()
    # concurrency_limit=1: one GovRAG index per process (one SQLite owner thread).
    ui.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
