# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2026-11-01

### Fixed
- **`pip install nepal-gov-agent[demo]` gave a demo that would not start.** The extra pinned
  `gradio<5`, and gradio 4.x fails on import with current `huggingface_hub` (`HfFolder` was
  removed). The extra is now `gradio>=5.0,<7`, and `demo/app.py` handles gradio 6's API changes
  (`theme` moved to `launch()`, `show_api` replaced by `footer_links`). Verified by launching the
  demo on gradio 5.50 and 6.31.

## [0.6.1] - 2026-10-31

### Removed
- `ragfallback` is no longer a dependency. The package never imported it; query rewriting comes
  from RAGNav's `QueryFallback`. Installs are lighter (no ragfallback or its extra LangChain
  dependencies pulled in). The README, demo and web page no longer list it.

## [0.6.0] - 2026-10-31

### Fixed

- **Rank order.** RAGNav's structure expansion returns blocks in document order, so
  `GovRAG.search()`, the extractive answer, `cited_block_ids` and benchmark Recall@1/@3 used
  "earliest page" instead of "best match". Score order is now restored from the retrieval trace.
  Synthetic-set Recall@1 on the seed corpus: 0.229 -> 0.490; Recall@3: 0.490 -> 0.604.
- **Repeated questions.** RAGNav's retrieval cache reported LOW confidence on every cache hit, so a
  repeated question re-ran QueryFallback and could flip `confidence` / `fallback_triggered`. GovRAG no
  longer uses that cache (the embedding cache is unchanged); `retrieval.db` is no longer created.
- `GovAgent.run()` validates the workflow before writing to session memory (an unknown workflow
  used to leave an orphan user turn).
- `run_benchmark(qa_pairs=[])` silently ran the smoke test under the caller's label; it now raises
  `ValueError`. `k_values` below 1 are rejected, and per-language recall prints `n/a` when the eval
  set has no queries in that language.
- `ingest_corpus()` raises `EmptyCorpusError` when no PDF yields text instead of building an empty
  index that fails later; files are processed in sorted order.
- `download_corpus()` writes through a `.part` file, so an interrupted download no longer leaves a
  truncated PDF that later runs skip; network/HTTP failures raise `CorpusDownloadError`.
- `OllamaClient` maps timeouts, HTTP errors (e.g. model not pulled) and malformed replies to
  `LLMUnavailableError` with actionable messages.
- Synthetic eval: OpenAI requests time out after 60 s (SDK default is 10 min); malformed resume
  lines are logged instead of silently dropped; PDFs are closed even if iteration stops early;
  `load_synthetic_qa()` reports the offending `file:line`.
- `web/app.py` keeps a reference to its background boot task so it cannot be garbage-collected.

### Added

- `nepal_gov_agent.exceptions`: `NepalGovAgentError`, `CorpusNotFoundError`, `EmptyCorpusError`,
  `CorpusDownloadError`, `LLMUnavailableError`, `MissingDependencyError`. Each also subclasses the
  built-in exception raised before (`FileNotFoundError`, `ValueError`, `RuntimeError`).
- `GovRAGConfig.fusion` (`"rrf"` default, unchanged behaviour; `"weighted"` blends normalised scores
  using `w_bm25` / `w_vec`, which RRF ignores) and validation of config values.
- `GovRAG.ask(llm=...)` now also uses that LLM for QueryFallback rewrites (previously always the
  offline echo stub).
- `OllamaClient(timeout=...)`; `generate_synthetic_qa(api_key=..., client=...)`;
  `nepal_gov_agent.eval.generate.main(argv)`. `GovAgentConfig` is exported from the package root.
- CodeQL workflow, pre-commit config, mypy in CI, coverage gate (95%), CI concurrency.
- Examples: `citizen_helpdesk.py`, `bring_your_own_corpus.py`, `cited_answers_llm.py` run offline on
  a generated mini corpus; `quickstart.py` and `benchmark_corpus.py` refreshed.

### Changed

- `download_corpus()` reports progress via the `nepal_gov_agent.corpus` logger instead of `print`
  (the `download` CLI command still shows it). `OllamaClient.embed()` raises `NotImplementedError`
  (a `RuntimeError` subclass, as before).
- CLI exits with status 2 and a one-line message on package errors (missing corpus, failed download).
- Ruff rule set broadened (SIM, C4, RUF, PTH, RET, PIE); test coverage 66% -> 96%.
- README rewritten; stale sample outputs replaced with a verified run.
- Removed `examples/nepali_query.py`, `agent_workflow.py`, `service_guide.py` (covered by the new
  examples).

## [0.5.0] - 2026-10-31

### Added

- `nepal-gov-agent download [--dest DIR] [--force]` CLI command and `nepal-gov-agent --version`.
- GitHub Actions CI (`tests.yml`): ruff lint + format check and pytest on Python 3.10–3.13.
- Offline test suite: tests build a small synthetic PDF corpus with PyMuPDF instead of relying on
  `Data/*.pdf` (which is no longer in git). Set `NEPAL_GOV_CORPUS_DIR` to run against the real seed
  corpus. New unit tests cover the CLI, config, query preprocessing, `OllamaClient` and
  `download_corpus` (HTTP mocked), synthetic-eval loading and report labelling, and cited answers.
- SECURITY.md, CODE_OF_CONDUCT.md, issue/PR templates, Dependabot, CITATION.cff.

### Changed

- **Python 3.10+ is now required** (3.9 dropped; 3.13 added).
- CLI `--corpus` now defaults to `./nepal_gov_data/` (where `download_corpus()` writes) instead of
  `Data/`, which no longer contains PDFs. Examples now call `download_corpus()` for the same reason.
- `__version__` is read from package metadata (`importlib.metadata`).
- Requires `ragnav>=0.5.1`, which fixes cited answers being rejected when the LLM writes
  `[[BLOCK_ID=<id>]]`.
- Packaging: `setuptools>=77`, PEP 639 license metadata, full project URLs, `dev` extra now has
  `build`/`twine` (and drops unused `black`; Ruff is the formatter).

### Fixed

- `GovRAG.ask(..., llm=...)` no longer swallows LLM / citation errors silently: it logs a warning
  and returns the extractive answer.
- `load_synthetic_qa()` / `run_synthetic_benchmark()` / `benchmark --synthetic` now find the
  bundled `eval_data/synthetic_qa_v1.jsonl` after `pip install` (it is installed under the
  environment's data directory, not the CWD).
- Benchmark `doc_hit` no longer counts a hit when a QA pair has no `expected_doc`.
- Synthetic-eval passage extraction imports `pymupdf` instead of the deprecated `fitz` alias.

## v0.4.1

- Pin `download_corpus()` URLs to the `v0.4.0` git tag so seed PDFs remain reachable after their removal from `main`.

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
