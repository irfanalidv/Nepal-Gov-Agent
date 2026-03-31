# Changelog

## v0.3.0

- Default embedding model is now `intfloat/multilingual-e5-small` for stronger Nepali / multilingual dense retrieval.
- Default hybrid weights are balanced: BM25 0.5 / vector 0.5 (previously 0.6 / 0.4).
- Added `preprocess_query()` — Unicode NFC normalization and optional stripping of common Nepali question suffixes before retrieval.
- `GovRAG` applies query preprocessing in `ask()` and `search()`, and clears the embedding cache automatically when `embedding_model` changes (tracked via `.embedding_model` marker in `cache_dir`). If SQLite `*.db` files exist but no marker is present (upgrades from older releases), the cache is cleared once so vectors are never mixed across models.
- Added `OllamaClient` for local answer synthesis via Ollama (e.g. `qwen2.5:7b`), using RAGNav inline-citation answering when passed as `llm=` to `GovRAG.ask(..., with_citations=True)`.
