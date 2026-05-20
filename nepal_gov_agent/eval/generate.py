"""
CLI entry point for synthetic QA generation.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m nepal_gov_agent.eval.generate \\
        --corpus Data/ \\
        --out eval_data/synthetic_qa_v1.jsonl \\
        --max-per-doc 20
"""

from __future__ import annotations

import argparse
import logging
import sys

from .synthetic import generate_synthetic_qa


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m nepal_gov_agent.eval.generate",
        description="Generate synthetic QA pairs from the Nepal gov corpus using OpenAI.",
    )
    parser.add_argument("--corpus", default="Data/", help="Corpus directory of PDFs")
    parser.add_argument(
        "--out", default="eval_data/synthetic_qa_v1.jsonl", help="Output JSONL path"
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI chat model")
    parser.add_argument(
        "--max-per-doc",
        type=int,
        default=20,
        help="Cap pairs per document for balance",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=40,
        help="Skip pages with fewer meaningful words",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output file and regenerate all pages",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(".env")
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    pairs = generate_synthetic_qa(
        corpus_dir=args.corpus,
        out_path=args.out,
        model=args.model,
        max_pairs_per_doc=args.max_per_doc,
        min_passage_words=args.min_words,
        resume=not args.no_resume,
    )

    by_doc: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for p in pairs:
        by_doc[p.expected_doc] = by_doc.get(p.expected_doc, 0) + 1
        by_lang[p.language] = by_lang.get(p.language, 0) + 1

    print("\n" + "=" * 60)
    print("Synthetic QA Generation — Summary")
    print("=" * 60)
    print(f"Total pairs:      {len(pairs)}")
    print(f"Output file:      {args.out}")
    print(f"Generator model:  {args.model}")
    print("\nBy document:")
    for doc, n in sorted(by_doc.items()):
        print(f"  {n:>4}  {doc}")
    print("\nBy language:")
    for lang, n in sorted(by_lang.items()):
        print(f"  {n:>4}  {lang}")
    print()
    print("⚠️  These pairs are LLM-generated and NOT human-validated.")
    print("    See eval_data/README.md for what that means.")
    print()


if __name__ == "__main__":
    main()
    sys.exit(0)
