"""
Nepal GovAgent — synthetic evaluation harness.

Generates LLM-authored QA pairs from the seed corpus, validates them against
source text, and reports retrieval metrics. Honest about what synthetic eval
is and isn't — see ``eval_data/README.md`` and ``docs/synthetic-eval.md``.
"""

from .synthetic import (
    SyntheticQAPair,
    generate_synthetic_qa,
    load_synthetic_qa,
)

__all__ = [
    "SyntheticQAPair",
    "generate_synthetic_qa",
    "load_synthetic_qa",
]
