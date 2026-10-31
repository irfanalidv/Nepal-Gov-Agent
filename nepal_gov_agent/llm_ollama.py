"""
Local LLM client via Ollama (fully offline on the host machine).

Recommended model: ``qwen2.5:7b`` — strong multilingual / Devanagari support
for a laptop-class setup.

Requires Ollama running locally (``ollama serve``) and a pulled model, e.g.:
``ollama pull qwen2.5:7b``
"""

from __future__ import annotations

from collections.abc import Iterable

import requests

from .exceptions import LLMUnavailableError


class OllamaClient:
    """Chat completions through Ollama's HTTP API.

    Implements the ``chat`` half of RAGNav's ``LLMClient`` protocol, so an instance
    can be passed straight to :meth:`GovRAG.ask` as ``llm=``.

    Args:
        model: Default Ollama model tag.
        base_url: Ollama server URL.
        timeout: Seconds to wait for a reply. Local 7B models on CPU can be slow,
            so the default is generous.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0,
    ) -> str:
        """Send ``messages`` to ``/api/chat`` and return the assistant text.

        Raises:
            LLMUnavailableError: Ollama is unreachable, timed out, returned an HTTP
                error (for example an unknown model), or sent a malformed reply.
                Subclass of ``RuntimeError``.
        """
        name = model or self.model
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": name,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            return str(data["message"]["content"])
        except requests.exceptions.ConnectionError as e:
            raise LLMUnavailableError(
                f"Ollama is not reachable at {self.base_url}. Start the daemon "
                f"(e.g. `ollama serve`) and pull a model: `ollama pull {name}`"
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMUnavailableError(
                f"Ollama did not answer within {self.timeout:g}s (model {name!r}). "
                "Raise OllamaClient(timeout=...) or use a smaller model."
            ) from e
        except requests.exceptions.HTTPError as e:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {e.response.status_code if e.response is not None else '?'}"
                f" for model {name!r}. Is it pulled? Try `ollama pull {name}`."
            ) from e
        except (KeyError, TypeError, ValueError) as e:
            raise LLMUnavailableError(f"Unexpected reply from Ollama: {e!r}") from e

    def embed(
        self,
        *,
        inputs: Iterable[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Not supported: GovRAG embeds with sentence-transformers instead."""
        raise NotImplementedError(
            "OllamaClient does not provide embeddings; GovRAG uses "
            "sentence-transformers for dense retrieval."
        )
