# Nepal GovAgent

> Reference RAG implementation over Nepal's policy and legal document corpus. Hybrid retrieval (BM25 + multilingual embeddings) with inline citations, agentic workflows, and offline-capable defaults. Built on [RAGNav](https://pypi.org/project/ragnav/), [ragfallback](https://pypi.org/project/ragfallback/), and [AgentEnsemble](https://pypi.org/project/agentensemble/).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/nepal-gov-agent.svg)](https://pypi.org/project/nepal-gov-agent/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/nepal-gov-agent?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/nepal-gov-agent)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-maintenance-blue.svg)](https://github.com/irfanalidv/Nepal-Gov-Agent#status)
[![Demo](https://img.shields.io/badge/demo-nepalgov.datacortex.in-indigo.svg)](https://nepalgov.datacortex.in)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irfanalidv/Nepal-Gov-Agent/blob/main/examples/Nepal_GovAgent_Demo.ipynb)

**Live demo:** [nepalgov.datacortex.in](https://nepalgov.datacortex.in) — try it against the seed corpus without installing anything.

---

## Status

**Maintenance.** This is a published reference implementation, not an actively roadmapped product. The goal of the project is to demonstrate end-to-end RAG against a real sovereign-AI corpus (Nepal's National AI Policy, Constitution, Digital Nepal Framework, etc.) using the author's open-source stack. PRs and issues are welcome; new feature work is on a best-effort basis.

If you want a production deployment with native-speaker NLP, real human-validated evaluation, and SLA support, that is consulting work — reach out via [datacortex.in](https://datacortex.in).

---

## Why this exists

Nepal has a digital government layer. The [Nagarik App](https://nagarikapp.gov.np/) has 1.5M+ registered citizens. Diyo AI has deployed Nepali-language chatbots for Lalitpur and Butwal municipalities. Paaila Technology has built Nepali NLU and TTS. Fusemachines is training hundreds of AI fellows.

The conversational layer exists. What is missing is a worked example of a retrieval-and-citation layer over the legal and policy corpus those systems sit on top of — one that runs offline, handles both Nepali and English queries against the same documents, and reports honest confidence and provenance for every answer.

This project is that worked example. It is not the entire answer to Nepal's [National AI Policy 2082](https://aiassociationnepal.org/national-artificial-intelligence-ai-policy-2082/), but it shows what one piece of the infrastructure could look like, with code you can read, modify, and deploy on a laptop.

---

## Install

```bash
pip install nepal-gov-agent
```

For answer generation with citations (optional):
```bash
pip install nepal-gov-agent[mistral]   # Mistral API
pip install nepal-gov-agent[ollama]    # Local Ollama
```

---

## Quick start

PDFs are **not** bundled on PyPI or in git. Fetch the seed set with `download_corpus()` (writes `./nepal_gov_data/`). Add your own PDFs to that folder or any path you pass to `GovRAG`.

```python
from nepal_gov_agent import GovRAG, GovRAGConfig, download_corpus

config = GovRAGConfig()  # offline by default; tune via GovRAGConfig fields
corpus_dir = download_corpus()
rag = GovRAG(corpus_dir=corpus_dir, config=config)
result = rag.ask("What is the vision of Nepal's National AI Policy?")

print(result.answer)
print(result.confidence)   # "high" | "medium" | "low"
for src in result.sources:
    print(src["doc"], "page", src["page"])
```

**Real output:**

```
[National AI Policy-Final_uxc94vg.pdf, p.1] Introduction
The vision of Nepal's National AI Policy is to build an ethical, safe, and
human-centric AI ecosystem that promotes inclusive and sustainable socio-economic
growth through responsible use of artificial intelligence...

---

[National AI Policy-Final_uxc94vg.pdf, p.2] Objectives
The mission focuses on maximising AI use for Nepal's socio-economic development...

high
National AI Policy-Final_uxc94vg.pdf page 1
National AI Policy-Final_uxc94vg.pdf page 2
dnf_jbji8eb.pdf page 4
```

---

## GovAgent workflows (Phase 2)

`GovAgent` layers [AgentEnsemble](https://github.com/irfanalidv/AgentEnsemble) pipeline orchestration on top of `GovRAG`, with SQLite session memory. Workflows: **`document_qa`** (default), **`service_guide`** (eligibility then procedure), **`corpus_search`** (raw blocks, no synthesis).

```python
import logging
from nepal_gov_agent import GovRAG, GovAgent, download_corpus

logging.basicConfig(level=logging.INFO)

corpus_dir = download_corpus()
rag = GovRAG(corpus_dir=corpus_dir)
agent = GovAgent(rag=rag, session_id="demo")

result = agent.run("What is the role of the National AI Centre?")
print(result.answer)
print("Confidence:", result.confidence)

guide = agent.run(
    "How do I apply for a citizenship certificate renewal in Nepal?",
    workflow="service_guide",
)
print(guide.answer[:500])
```

CLI:

```bash
nepal-gov-agent agent "How do I renew my citizenship?"
nepal-gov-agent agent "How do I get a driving licence?" --workflow service_guide
nepal-gov-agent agent "AI policy infrastructure" --workflow corpus_search
nepal-gov-agent agent "..." --session my_session_id
```

Examples: `examples/agent_workflow.py`, `examples/service_guide.py`.

---

## Use cases

Examples below assume you have run `corpus_dir = download_corpus()` once (→ `./nepal_gov_data/`).

### 1. Query the Nepal AI Policy (English)

```python
from nepal_gov_agent import GovRAG, download_corpus

corpus_dir = download_corpus()
rag = GovRAG(corpus_dir=corpus_dir)

result = rag.ask("What is the role of the National AI Centre?")
print(result.answer)
print("Confidence:", result.confidence)
```

**Real output:**
```
[National AI Policy-Final_uxc94vg.pdf, p.18] National AI Centre
The National AI Centre shall serve as the Secretariat of the AI Regulation
Council. It will promote, encourage, and regulate AI study, research,
development, and application. The Centre is tasked with conducting research
and development in the AI technology sector and supporting the capacity
development of local researchers and institutions...

Confidence: high
```

---

### 2. Nepali language query

```python
result = rag.ask("नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?")

print(result.answer)
print("Confidence:", result.confidence)
print("Fallback triggered:", result.fallback_triggered)
```

**Real output:**
```
[National AI Policy-Final_uxc94vg.pdf, p.1] Introduction
Nepal's AI Policy aims to build a reliable AI ecosystem, boost economic
productivity, improve public services, and strengthen governance. The
objectives include building infrastructure, promoting research, and
ensuring ethical and secure AI deployment...

Confidence: medium
Fallback triggered: False
```

---

### 3. Constitution query

```python
result = rag.ask("What does the Constitution say about fundamental rights?")

print(result.answer[:600])
print("\nSources:")
for src in result.sources[:3]:
    print(f"  {src['doc']} — page {src['page']}")
    if src["heading"]:
        print(f"  Section: {src['heading']}")
```

**Real output:**
```
[Constitution of Nepal (2nd amd. English)_xf33zb3.pdf, p.12] Part 3 — Fundamental Rights
Every citizen shall have the following rights: Right to equality before
law. No discrimination shall be made against any citizen in the application
of general laws on grounds of origin, religion, race, caste, tribe, sex,
economic condition, language, region, ideology or on similar other grounds...

Sources:
  Constitution of Nepal (2nd amd. English)_xf33zb3.pdf — page 12
  Section: Part 3 — Fundamental Rights
  Constitution of Nepal (2nd amd. English)_xf33zb3.pdf — page 13
  Constitution of Nepal (2nd amd. English)_xf33zb3.pdf — page 14
```

---

### 4. Raw search (no answer generation)

```python
blocks = rag.search("National AI Centre secretariat", k=5)

for b in blocks:
    print(b["doc_id"])
    print(b["title"])
    print(b["content"][:200])
    print("---")
```

**Real output:**
```
pdf:National AI Policy-Final_uxc94vg.pdf
None
10.2 National AI Centre
The National AI Centre shall serve as the Secretariat of the AI Regulation
Council. It will promote, encourage, and regulate AI study, research,
development and application...
---
pdf:National AI Policy-Final_uxc94vg.pdf
None
10.1 AI Regulation Council
The AI Regulation Council shall meet at least twice a year. The National
AI Centre shall serve as its Secretariat...
---
```

---

### 5. With LLM for cited answers (Mistral)

```python
import os
from ragnav.llm.mistral import MistralClient
from nepal_gov_agent import GovRAG, download_corpus

os.environ["MISTRAL_API_KEY"] = "your_key_here"  # or load from .env

corpus_dir = download_corpus()
rag = GovRAG(corpus_dir=corpus_dir)
llm = MistralClient()

result = rag.ask(
    "How many AI professionals does Nepal aim to train?",
    llm=llm,
    with_citations=True,
)
print(result.answer)
```

**Real output (with citations):**
```
Nepal aims to train at least 5,000 skilled AI professionals within five
years [[pdf:National AI Policy-Final_uxc94vg.pdf#p3]]. This target is
part of a broader capacity-building initiative that includes integrating
AI curricula from school to university level [[pdf:National AI Policy-Final_uxc94vg.pdf#p4]].
```

---

### 6. Run the benchmark

```python
from nepal_gov_agent import GovRAG, run_benchmark

rag = GovRAG(corpus_dir="./nepal_gov_data/")
results = run_benchmark(rag, verbose=True)
print(results.report())
```

**Real output** (from a local corpus with five seed PDFs via `download_corpus()` — run `run_benchmark(rag)` on your `corpus_dir` for live numbers):
```
  ✓ [english] What is the vision of Nepal's National AI Policy?...
  ✓ [english] What are the four pillars of Nepal's AI readiness?...
  ✓ [english] What is the role of the National AI Centre?...
  ✓ [english] How many AI professionals does Nepal aim to train?...
  ✓ [english] What does the Constitution say about fundamental rights...
  ✓ [english] What is Digital Nepal Framework 2.0?...
  ✓ [nepali] नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?...

============================================================
Nepal GovAgent Benchmark Results
============================================================
Total queries:      7
Recall@1:           0.571
Recall@3:           0.857
Recall@5:           1.000
Keyword hit rate:   1.000
Doc hit rate:       1.000
Nepali recall@3:    1.000
English recall@3:   0.833
============================================================
```

> **Note:** These numbers measure **retrieval**, not generated answer quality. Recall@3 = 0.857 means the expected keywords appeared in the top 3 retrieved blocks for 6 of 7 queries. Doc hit rate = 1.000 means the expected source PDF appeared in the top‑5 hits for every query (harness normalizes the `pdf:` doc id prefix). See [Benchmark: retrieval not answer quality](#benchmark-retrieval-not-answer-quality) below.

#### Synthetic eval (v0.4.0+)

The 7-question set above is a smoke test. For a larger eval, `nepal_gov_agent.eval` can auto-generate QA pairs from each page of the corpus using OpenAI, validating that expected keywords appear verbatim in the source passage before saving the pair.

```bash
pip install 'nepal-gov-agent[eval]'
export OPENAI_API_KEY=sk-...

# Generate ~100-200 pairs across the seed corpus (idempotent)
python -m nepal_gov_agent.eval.generate --corpus ./nepal_gov_data/ --max-per-doc 20 --verbose

# Run benchmark against the generated set
nepal-gov-agent benchmark --corpus ./nepal_gov_data/ --synthetic
```

The report header explicitly labels the result as `SYNTHETIC eval — LLM-generated, NOT human-validated`, and ships with a disclaimer block. **These numbers are a regression signal, not ground truth.** Anyone reporting them without the disclaimer is misrepresenting what they are. See [`eval_data/README.md`](eval_data/README.md) for what synthetic eval is, isn't, and what real evaluation would look like.

---

### 7. CLI

```bash
# Ask in English
nepal-gov-agent ask "What is Nepal's National AI Policy?"

# Ask in Nepali
nepal-gov-agent ask "नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?"

# Custom corpus folder
nepal-gov-agent ask "What are fundamental rights?" --corpus /path/to/docs/

# Retrieve more blocks
nepal-gov-agent ask "AI infrastructure" --k 10

# Run benchmark (after download_corpus() → ./nepal_gov_data/)
nepal-gov-agent benchmark --corpus ./nepal_gov_data/
nepal-gov-agent benchmark --corpus ./nepal_gov_data/

# Run synthetic benchmark (generate first with python -m nepal_gov_agent.eval.generate)
nepal-gov-agent benchmark --corpus ./nepal_gov_data/ --synthetic

# Show corpus stats
nepal-gov-agent stats

# GovAgent (Phase 2): document_qa (default), service_guide, corpus_search
nepal-gov-agent agent "How do I renew my citizenship?"
nepal-gov-agent agent "citizenship renewal" --workflow service_guide
nepal-gov-agent agent "National AI Centre" --workflow corpus_search --session my_session
```

**Real CLI output (`ask`):**
```
============================================================
ANSWER
============================================================
[National AI Policy-Final_uxc94vg.pdf, p.1] Introduction
The vision of Nepal's National AI Policy is to build an ethical, safe,
and human-centric AI ecosystem...

============================================================
SOURCES
============================================================
1. National AI Policy-Final_uxc94vg.pdf (page 1)
   Excerpt: The vision of Nepal's National AI Policy is to build an ethical...
2. National AI Policy-Final_uxc94vg.pdf (page 2)
   Excerpt: The mission focuses on maximising AI use for Nepal's socio-econ...
3. dnf_jbji8eb.pdf (page 4)
   Excerpt: Digital Nepal Framework 2.0 builds upon the comprehensive vision...

Confidence: high
```

---

### 8. Corpus statistics

```python
import logging

from nepal_gov_agent import GovRAG

logging.basicConfig(level=logging.WARNING)

rag = GovRAG(corpus_dir="./nepal_gov_data/")
print(rag.stats)
```

**Real output:**
```python
{
    "documents": 6,
    "blocks": 856,
    "corpus_dir": "./nepal_gov_data/",
    "offline": True
}
```

---

### 9. Custom config

```python
from nepal_gov_agent import GovRAG, GovRAGConfig

config = GovRAGConfig(
    w_bm25=0.7,              # More weight on keyword search
    w_vec=0.3,               # Less weight on semantic search
    k_final=12,              # Retrieve more blocks
    max_fallback_attempts=5, # More retries on low confidence
    cache_dir=".my_cache",
    embedding_model="intfloat/multilingual-e5-small",
)

rag = GovRAG(corpus_dir="./nepal_gov_data/", config=config)
result = rag.ask("What is the Digital Nepal Framework?")
```

---

## Corpus

### Seed corpus (`download_corpus()`)

The wheel and git repo do **not** bundle PDFs. `download_corpus()` pulls **five** public policy/gazette files into `./nepal_gov_data/`:

| Document | Language |
|---|---|
| National AI Policy 2082 | English |
| Constitution of Nepal (2nd amendment) | English |
| Digital Nepal Framework 2.0 | English |
| प्रतिनिधि सभा सदस्य निर्वाचन अध्यादेश २०८२ | Nepali |
| मानव अधिकार पुरस्कार कोष सञ्चालन नियमावली २०७५ | Nepali |

It **omits** *Legal Maxims* (`1714977234_32.pdf`) because that reference volume is very large and often dominates hybrid retrieval over shorter Nepali ordinances. Add that file yourself if you need it for local experiments.

```python
from nepal_gov_agent import download_corpus, GovRAG

corpus_dir = download_corpus()  # → ./nepal_gov_data/ (absolute path returned)
rag = GovRAG(corpus_dir=corpus_dir)
```

### Bring your own corpus

Point `GovRAG` at any folder of Nepal government PDFs — ministry circulars, municipal SOPs, provincial guidelines, land records, anything you are allowed to use:

```python
rag = GovRAG(corpus_dir="./my_ministry_docs/")
```

No extra configuration. Put PDFs in a single folder (not subfolders) and pass that path as `corpus_dir`.

**Contributing:** If you have publicly available Nepal government PDFs, open an issue with links or a PR describing sources — expanding the corpus is the highest-priority contribution this project needs.

---

## Under the hood

```
Nepal GovAgent
│
├── RAG Layer          →  ragnav==0.3.0
│   ├── PDF ingestion (PyMuPDF, page-level blocks)
│   ├── BM25 index (rank-bm25)
│   ├── Vector index (sentence-transformers, multilingual-e5-small by default)
│   ├── Hybrid retrieval (BM25 0.5 + vector 0.5 by default, RRF fusion)
│   ├── Structure expansion (parent/child block hierarchy)
│   ├── QueryFallback for low-confidence query retries
│   ├── Inline citation enforcement
│   └── SQLite embedding + retrieval cache (offline)
│
├── Reliability Layer  →  ragfallback==2.0.2
│   └── Metrics and adaptive utilities (PyPI dependency; GovRAG benchmark uses keyword / Recall@k on retrieved blocks)
│
└── Agent Layer        →  agentensemble==0.3.0 (Phase 2)
    └── Multi-step workflows, SQLite session memory (planned)
```

**Key design decisions:**
- Default hybrid weights 0.5 / 0.5: multilingual embeddings help Nepali queries; BM25 still anchors legal keywords
- Offline by default: `intfloat/multilingual-e5-small` runs on CPU with no API key needed
- SQLite cache: embeddings cached locally — second run is instant
- Fallback only on `ConfidenceLevel.LOW`: avoids unnecessary LLM calls on already-confident retrievals

---

## Benchmark: retrieval, not answer quality

The built-in benchmark measures **retrieval quality** — whether the right content appears in the top-k retrieved blocks. It does not evaluate answer accuracy, fluency, or factual correctness.

When presenting results to stakeholders:

> "Recall@3 = 0.857 means that for 6 out of 7 test queries, the relevant section of the government document appeared in the top 3 retrieved chunks."

This is a meaningful signal for infrastructure quality. Answer quality evaluation (LLM-as-judge, BLEU/ROUGE) is planned for a later phase.

---

## The gap this fills

| Layer | Who's doing it | What's missing |
|---|---|---|
| Nepali NLP / NLU | Paaila Technology, Diyo AI | — |
| Government chatbots | Diyo AI (Muna), Fusemachines | — |
| AI training / fellowships | Fusemachines | — |
| **Legal RAG with citation** | **Nobody** | **← This project** |
| **Adaptive retrieval fallback** | **Nobody** | **← This project** |
| **Offline-capable full stack** | **Nobody** | **← This project** |

---

## Nepal context

- **August 2025** — Nepal Cabinet approved [National AI Policy 2082](https://aiassociationnepal.org/national-artificial-intelligence-ai-policy-2082/)
- **November 2025** — AIAN + Embassy of India hosted "AI for Inclusive Growth: Building Nepal's AI Ready Future" (400+ participants)
- **February 2026** — [World Bank approved $50M](https://coingeek.com/nepal-secures-50m-for-digitalization-approved-by-world-bank/) Nepal Digital Transformation Project
- **March 2026** — [Nagarik App at 1.5M+ users](https://en.wikipedia.org/wiki/Nagarik_App); backend intelligence layer remains an open gap

---

## Release history

**Current release:** `0.4.0` on PyPI — **18** tests passing; seed corpus via `download_corpus()` (opt-in, five PDFs).

- **0.4.0** — Polish + honesty pass. Synthetic eval harness with explicit "LLM-generated, not human-validated" labelling at every layer (report header, dataset README, CHANGELOG). Hosted Gradio demo at [nepalgov.datacortex.in](https://nepalgov.datacortex.in). Project reframed as a reference implementation; status set to maintenance.
- **0.3.0** — Multilingual default embeddings (`multilingual-e5-small`), balanced BM25/vector weights, Nepali query preprocessing, Ollama client for local synthesis.
- **0.2.0** — `GovAgent` class with `document_qa`, `service_guide`, `corpus_search` workflows via AgentEnsemble pipeline; SQLite session memory.
- **0.1.x** — `GovRAG` core: hybrid retrieval, adaptive fallback, inline citations, benchmark harness.

## Future work (community)

This project is in maintenance. The directions below are open if a contributor wants to pick one up — no committed timeline from the author:

- Native-speaker review of the synthetic eval set; curated human-validated subset.
- Real Nepali NLP stack (IndicNLP or Stanza) for tokenization and transliteration handling, replacing the current minimal suffix-stripping.
- Corpus expansion to ministry circulars 2080–2082 and a structured index over amendments.
- Voice workflow integration (Bhashini / Bolna) for Nepali-language phone interfaces.
- Multi-hop reasoning across documents (e.g. "which articles of the Constitution conflict with this draft ordinance?").

Open an issue before starting on anything large so we can coordinate scope.

---

## Scope and Limitations

Nepal GovAgent is a **research prototype**. It is not production-ready for government deployment.

The benchmark measures **retrieval quality** — whether the right content appears in the top-k retrieved blocks. It does not measure answer safety, factual correctness, or suitability for official use.

Prerequisites for any .gov deployment that this project does not provide:

- Security audit and penetration testing
- Data sovereignty controls and legal framework
- Human oversight layer — every answer reviewed before action
- Government approval and integration with live systems

This project establishes the retrieval foundation layer. The trust, oversight, and sovereignty layers are separate concerns that must be built on top before any production use.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Highest priority:** Nepal government document corpus. If you have publicly available PDFs — ministry circulars, SOPs, municipality guidelines — open an issue with links or sources.

---

## Built on

| Library | Role | Version |
|---|---|---|
| [RAGNav](https://github.com/irfanalidv/RAGNav) | Retrieval, citations, PDF ingestion | `>=0.3.0` |
| [ragfallback](https://github.com/irfanalidv/ragfallback) | Reliability, fallback, diagnostics | `>=2.0.2` |
| [AgentEnsemble](https://github.com/irfanalidv/AgentEnsemble) | Orchestration, memory, planning | `>=0.3.0` |

All three are open-source, MIT licensed, and independently usable on PyPI.

---

## About

Built by [Irfan Ali](https://www.linkedin.com/in/irfanalidv/) — [GitHub](https://github.com/irfanalidv) | Founder, [DataCortex IQ](https://www.datacortex.in/) | 7+ years in LLM engineering, RAG pipelines, and agentic AI | 11 open-source Python libraries on PyPI | M.Sc. Data Science & AI, IISER Tirupati.

Built in the spirit of AIAN's four pillars: **Data. Infrastructure. Policy. Resources.**

---

## License

MIT — free for government, private sector, and academic use. No strings.

---

*Working on Nepal's AI infrastructure layer? Open an issue or reach out directly.*
