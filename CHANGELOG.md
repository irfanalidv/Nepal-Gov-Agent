# Changelog

## v0.4.0

This release is a polish + honesty pass. No new agentic capability — the goal is to make the existing functionality credible enough to point a stranger at without caveats in person.

- **Synthetic evaluation harness.** New `nepal_gov_agent.eval` package generates QA pairs from each page of the seed corpus using OpenAI (default `gpt-4.1-mini`), validates that expected keywords appear verbatim in the source passage, and persists to JSONL. Idempotent — already-covered pages are skipped on re-run. Run with `python -m nepal_gov_agent.eval.generate` or `from nepal_gov_agent import generate_synthetic_qa`.
- **Honesty contract on benchmark results.** `BenchmarkResult` now carries `eval_kind`, `is_synthetic`, and `generator_model`. Reports rendered from synthetic eval sets are explicitly labeled `SYNTHETIC eval — LLM-generated, NOT human-validated` in the header, with a disclaimer block in the footer. The hand-curated 7-question set is now labeled as a smoke test rather than a benchmark, which is what it always was.
- **`run_synthetic_benchmark()`** convenience wrapper loads the JSONL file and runs the eval with the correct provenance flags pre-set.
- **`nepal-gov-agent benchmark --synthetic`** CLI flag runs the synthetic eval end-to-end.
- **`eval_data/README.md`** ships with the package and explains exactly what the synthetic set is, what it isn't, and what real human-validated evaluation would look like. The disclaimer travels with the numbers wherever they go.
- **First-run finding: meaningful English/Nepali retrieval gap on this corpus.** The first run of the synthetic eval surfaces a substantial drop in Recall@3 from English to Nepali queries against the same documents and the same pipeline. The most likely first-order cause is the multilingual embedding layer (`multilingual-e5-small`) underperforming on Devanagari policy register rather than the retrieval architecture itself, but some fraction of the gap is plausibly a test-as-artefact effect from LLM-translated Nepali questions. The decomposition between "model weakness" and "eval-set noise" requires native-speaker review of the Nepali side. See [docs/synthetic-eval.md](docs/synthetic-eval.md) for the full reading.
- **New optional dependency groups:** `[eval]` (adds `openai`), `[demo]` (adds `gradio` for the hosted demo at nepalgov.datacortex.in).
- **Hosted demo.** `nepalgov.datacortex.in` runs the library against the seed corpus so anyone can try it without `pip install`. Source in `demo/`.
- **Project framing.** Description, README, and docs now position the project as a reference implementation rather than a product, which is what it is. Closing the chapter cleanly so contributors aren't misled about the maintenance commitment.

### Breaking changes

- `__version__` bumped to `0.4.0`.
- `BenchmarkResult` adds new fields with safe defaults; existing code that builds the dataclass positionally will still work.

## v0.3.0

- Default embedding model is now `intfloat/multilingual-e5-small` for stronger Nepali / multilingual dense retrieval.
- Default hybrid weights are balanced: BM25 0.5 / vector 0.5 (previously 0.6 / 0.4).
- Added `preprocess_query()` — Unicode NFC normalization and optional stripping of common Nepali question suffixes before retrieval.
- `GovRAG` applies query preprocessing in `ask()` and `search()`, and clears the embedding cache automatically when `embedding_model` changes (tracked via `.embedding_model` marker in `cache_dir`). If SQLite `*.db` files exist but no marker is present (upgrades from older releases), the cache is cleared once so vectors are never mixed across models.
- Added `OllamaClient` for local answer synthesis via Ollama (e.g. `qwen2.5:7b`), using RAGNav inline-citation answering when passed as `llm=` to `GovRAG.ask(..., with_citations=True)`.
