"""
Local LLM client via Ollama (fully offline on the host machine).

Recommended model: ``qwen2.5:7b`` — strong multilingual / Devanagari support
for a laptop-class setup.

Requires Ollama running locally (``ollama serve``) and a pulled model, e.g.:
``ollama pull qwen2.5:7b``
"""

from __future__ import annotations

from typing import Iterable, Optional

import requests


class OllamaClient:
    """Chat completions through Ollama's HTTP API."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0,
    ) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            return str(data["message"]["content"])
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                "Ollama is not reachable. Start the daemon (e.g. `ollama serve`) "
                "and pull a model: `ollama pull qwen2.5:7b`"
            ) from e

    def embed(
        self,
        *,
        inputs: Iterable[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        raise RuntimeError(
            "OllamaClient does not provide embeddings; GovRAG uses "
            "sentence-transformers for dense retrieval."
        )
