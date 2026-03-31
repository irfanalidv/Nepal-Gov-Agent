"""
CLI for Nepal GovAgent.

Usage:
    nepal-gov-agent ask "What is Nepal's AI policy?"
    nepal-gov-agent ask "नागरिकता नवीकरण गर्न के चाहिन्छ?"
    nepal-gov-agent agent "How do I renew my citizenship?"
    nepal-gov-agent agent "..." --workflow service_guide --session my_session_id
    nepal-gov-agent benchmark
    nepal-gov-agent stats
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nepal-gov-agent",
        description="Agentic AI for Nepal government documents",
    )
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", help="Question in Nepali or English")
    ask_parser.add_argument("--corpus", default="Data/", help="Corpus directory")
    ask_parser.add_argument("--k", type=int, default=5, help="Number of blocks to retrieve")

    bench_parser = subparsers.add_parser("benchmark", help="Run retrieval benchmark")
    bench_parser.add_argument("--corpus", default="Data/", help="Corpus directory")

    stats_parser = subparsers.add_parser("stats", help="Show corpus statistics")
    stats_parser.add_argument("--corpus", default="Data/", help="Corpus directory")

    agent_parser = subparsers.add_parser("agent", help="Run GovAgent workflow")
    agent_parser.add_argument("query", help="Question or search text")
    agent_parser.add_argument(
        "--workflow",
        default="document_qa",
        choices=("document_qa", "service_guide", "corpus_search"),
        help="document_qa | service_guide | corpus_search",
    )
    agent_parser.add_argument("--session", default="default", help="Session id for memory")
    agent_parser.add_argument("--corpus", default="Data/", help="Corpus directory")
    agent_parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="k_final for QA / k for corpus_search (defaults: RAG config / 5)",
    )

    args = parser.parse_args()

    if args.command == "ask":
        logging.basicConfig(level=logging.INFO)
        from .rag import GovRAG

        rag = GovRAG(corpus_dir=args.corpus)
        result = rag.ask(args.query, k_final=args.k)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result.answer)
        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)
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

    elif args.command == "benchmark":
        logging.basicConfig(level=logging.INFO)
        from .benchmark import run_benchmark
        from .rag import GovRAG

        print("Initializing Nepal GovAgent...")
        rag = GovRAG(corpus_dir=args.corpus)
        print("\nRunning benchmark...\n")
        run_benchmark(rag, verbose=True)

    elif args.command == "stats":
        logging.basicConfig(level=logging.WARNING)
        from .rag import GovRAG

        rag = GovRAG(corpus_dir=args.corpus)
        stats = rag.stats
        print(f"Documents: {stats['documents']}")
        print(f"Blocks:    {stats['blocks']}")
        print(f"Corpus:    {stats['corpus_dir']}")
        print(f"Offline:   {stats['offline']}")

    elif args.command == "agent":
        logging.basicConfig(level=logging.INFO)
        from .agent import GovAgent
        from .rag import GovRAG

        rag = GovRAG(corpus_dir=args.corpus)
        gov_agent = GovAgent(rag=rag, session_id=args.session)
        result = gov_agent.run(args.query, workflow=args.workflow, k_final=args.k)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result.answer)
        print("\n" + "=" * 60)
        print("META")
        print("=" * 60)
        print("Workflow:   %s" % result.workflow)
        print("Confidence: %s" % result.confidence)
        print("Steps:      %d" % len(result.steps))
        print("Sources:    %d" % len(result.sources))
        for i, src in enumerate(result.sources[:5], 1):
            doc = src.get("doc", "?")
            page = src.get("page", "?")
            print("  %d. %s (page %s)" % (i, doc, page))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
