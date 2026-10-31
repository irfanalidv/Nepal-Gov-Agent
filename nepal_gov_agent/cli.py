"""
CLI for Nepal GovAgent.

Usage:
    nepal-gov-agent ask "What is Nepal's AI policy?"
    nepal-gov-agent ask "नागरिकता नवीकरण गर्न के चाहिन्छ?"
    nepal-gov-agent agent "How do I renew my citizenship?"
    nepal-gov-agent agent "..." --workflow service_guide --session my_session_id
    nepal-gov-agent download
    nepal-gov-agent benchmark
    nepal-gov-agent stats
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

from .exceptions import NepalGovAgentError

if TYPE_CHECKING:
    from .rag import GovRAG

# Same folder ``download_corpus()`` / ``nepal-gov-agent download`` write to.
DEFAULT_CORPUS_DIR = "./nepal_gov_data/"
CORPUS_HELP = "Folder of PDFs (default: ./nepal_gov_data/, filled by `nepal-gov-agent download`)"
RULE = "=" * 60


def _build_parser(version: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nepal-gov-agent",
        description="Agentic AI for Nepal government documents",
    )
    parser.add_argument("--version", action="version", version=f"nepal-gov-agent {version}")
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", help="Question in Nepali or English")
    ask_parser.add_argument("--corpus", default=DEFAULT_CORPUS_DIR, help=CORPUS_HELP)
    ask_parser.add_argument("--k", type=int, default=5, help="Number of blocks to retrieve")

    bench_parser = subparsers.add_parser("benchmark", help="Run retrieval benchmark")
    bench_parser.add_argument("--corpus", default=DEFAULT_CORPUS_DIR, help=CORPUS_HELP)
    bench_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use LLM-generated synthetic eval set instead of the 7-question smoke test",
    )
    bench_parser.add_argument(
        "--synthetic-path",
        default="eval_data/synthetic_qa_v1.jsonl",
        help="Path to synthetic QA JSONL (used with --synthetic)",
    )

    dl_parser = subparsers.add_parser("download", help="Download the seed corpus PDFs")
    dl_parser.add_argument("--dest", default=DEFAULT_CORPUS_DIR, help="Destination folder")
    dl_parser.add_argument("--force", action="store_true", help="Re-download existing files")

    stats_parser = subparsers.add_parser("stats", help="Show corpus statistics")
    stats_parser.add_argument("--corpus", default=DEFAULT_CORPUS_DIR, help=CORPUS_HELP)

    agent_parser = subparsers.add_parser("agent", help="Run GovAgent workflow")
    agent_parser.add_argument("query", help="Question or search text")
    agent_parser.add_argument(
        "--workflow",
        default="document_qa",
        choices=("document_qa", "service_guide", "corpus_search"),
        help="document_qa | service_guide | corpus_search",
    )
    agent_parser.add_argument("--session", default="default", help="Session id for memory")
    agent_parser.add_argument("--corpus", default=DEFAULT_CORPUS_DIR, help=CORPUS_HELP)
    agent_parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="k_final for QA / k for corpus_search (defaults: RAG config / 5)",
    )
    return parser


def _load_rag(corpus: str) -> GovRAG:
    from .rag import GovRAG

    return GovRAG(corpus_dir=corpus)


def _cmd_ask(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    result = _load_rag(args.corpus).ask(args.query, k_final=args.k)

    print(f"\n{RULE}\nANSWER\n{RULE}")
    print(result.answer)
    print(f"\n{RULE}\nSOURCES\n{RULE}")
    for i, src in enumerate(result.sources[:5], 1):
        print(f"{i}. {src['doc']} (page {src['page']})")
        if src.get("heading"):
            print(f"   Section: {src['heading']}")
        ex = src.get("excerpt") or ""
        if ex:
            tail = "..." if len(ex) > 100 else ""
            print(f"   Excerpt: {ex[:100]}{tail}")
    print(f"\nConfidence: {result.confidence}")
    if result.fallback_triggered:
        print(f"Note: fallback triggered — used query: '{result.query_used}'")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    from .benchmark import run_benchmark, run_synthetic_benchmark

    print("Initializing Nepal GovAgent...")
    rag = _load_rag(args.corpus)
    if args.synthetic:
        print("\nRunning SYNTHETIC benchmark (LLM-generated eval)...\n")
        result = run_synthetic_benchmark(rag, path=args.synthetic_path, verbose=True)
    else:
        print("\nRunning smoke-test benchmark (7 hand-curated questions)...\n")
        result = run_benchmark(rag, verbose=True)
    print("\n" + result.report())


def _cmd_download(args: argparse.Namespace) -> None:
    # download_corpus reports progress through logging; show it on the terminal.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .corpus import download_corpus

    download_corpus(dest_dir=args.dest, force=args.force)


def _cmd_stats(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING)
    stats = _load_rag(args.corpus).stats
    print(f"Documents: {stats['documents']}")
    print(f"Blocks:    {stats['blocks']}")
    print(f"Corpus:    {stats['corpus_dir']}")
    print(f"Offline:   {stats['offline']}")


def _cmd_agent(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    from .agent import GovAgent

    gov_agent = GovAgent(rag=_load_rag(args.corpus), session_id=args.session)
    result = gov_agent.run(args.query, workflow=args.workflow, k_final=args.k)

    print(f"\n{RULE}\nANSWER\n{RULE}")
    print(result.answer)
    print(f"\n{RULE}\nMETA\n{RULE}")
    print(f"Workflow:   {result.workflow}")
    print(f"Confidence: {result.confidence}")
    print(f"Steps:      {len(result.steps)}")
    print(f"Sources:    {len(result.sources)}")
    for i, src in enumerate(result.sources[:5], 1):
        print(f"  {i}. {src.get('doc', '?')} (page {src.get('page', '?')})")


_COMMANDS = {
    "ask": _cmd_ask,
    "benchmark": _cmd_benchmark,
    "download": _cmd_download,
    "stats": _cmd_stats,
    "agent": _cmd_agent,
}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``nepal-gov-agent`` console script.

    Exits with status 1 when no sub-command is given and status 2 on a known
    package error (missing corpus, download failure, ...), printing a one-line
    message instead of a traceback.
    """
    from . import __version__

    parser = _build_parser(__version__)
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    try:
        handler(args)
    except NepalGovAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
