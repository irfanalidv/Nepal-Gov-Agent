"""Exception hierarchy for Nepal GovAgent.

Every exception subclasses both :class:`NepalGovAgentError` (so callers can catch
anything raised by this package in one place) and the built-in type that earlier
releases raised (so existing ``except FileNotFoundError`` / ``except ValueError`` /
``except RuntimeError`` handlers keep working).
"""

from __future__ import annotations


class NepalGovAgentError(Exception):
    """Base class for all errors raised by ``nepal_gov_agent``."""


class CorpusNotFoundError(NepalGovAgentError, FileNotFoundError):
    """The corpus directory does not exist."""


class EmptyCorpusError(NepalGovAgentError, ValueError):
    """The corpus directory has no usable PDFs (none found, or none could be parsed)."""


class CorpusDownloadError(NepalGovAgentError, RuntimeError):
    """A seed-corpus PDF could not be downloaded."""


class LLMUnavailableError(NepalGovAgentError, RuntimeError):
    """A local or remote LLM backend could not be reached or returned an unusable reply."""


class MissingDependencyError(NepalGovAgentError, RuntimeError):
    """An optional dependency (extra) required for this feature is not installed."""


__all__ = [
    "CorpusDownloadError",
    "CorpusNotFoundError",
    "EmptyCorpusError",
    "LLMUnavailableError",
    "MissingDependencyError",
    "NepalGovAgentError",
]
