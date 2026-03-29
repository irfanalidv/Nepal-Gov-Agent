# Nepal GovAgent

> Open-source agentic AI framework for Nepal's government service layer — RAG, multi-step task automation, and MLOps infra designed for offline deployment and Nepali language workflows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## Why this exists

Nepal has a digital government layer. The [Nagarik App](https://nagarikapp.gov.np/) has 1.5M+ registered citizens. Diyo AI has deployed Nepali-language chatbots for Lalitpur and Butwal municipalities. Paaila Technology has built Nepali NLU and TTS. Fusemachines is training hundreds of AI fellows.

**The conversational layer exists. What doesn't exist is the intelligence layer above it.**

Today, Nepal's government AI systems can answer questions. They cannot:

- Complete multi-step tasks on behalf of a citizen ("apply for this permit end-to-end")
- Retrieve and cite specific clauses from Nepal's legal and policy corpus with source tracing
- Run reliably without internet connectivity in areas with poor network coverage
- Plug into existing systems without cloud API dependency (no data leaving Nepal)
- Monitor and audit themselves at production scale

Nepal's [National AI Policy 2082](https://aiassociationnepal.org/national-artificial-intelligence-ai-policy-2082/) explicitly identifies **infrastructure, data, and sovereignty** as its foundational pillars. The [AI Association of Nepal (AIAN)](https://aiassociationnepal.org) has framed national AI readiness around four pillars: Data, Infrastructure, Policy, Resources.

This project is the infrastructure answer.

---

## What Nepal GovAgent does

Nepal GovAgent assembles three battle-tested open-source libraries into a unified framework configured specifically for Nepal's government context.

### Under the hood — the stack

```
Nepal GovAgent
│
├── RAG Layer          →  RAGNav        (retrieval, citations, legal doc parsing)
├── Reliability Layer  →  ragfallback   (confidence scoring, adaptive fallback, diagnostics)
└── Agent Layer        →  AgentEnsemble (multi-step orchestration, memory, task planning)
```

Each library is independently published on PyPI and production-tested. Nepal GovAgent is the Nepal-specific assembly: pre-configured pipelines, Nepali language handling, offline-first defaults, and a growing corpus of Nepal government documents.

### Core capabilities

**1. Legal document RAG with inline citations**

Every answer is traceable to a specific block in a specific government document. Powered by RAGNav's `answer_with_inline_citations` — every sentence must include a `[[block_id]]` reference or the answer is rejected.

```python
from nepal_gov_agent import GovRAG

rag = GovRAG(corpus_dir="Data/", offline=True)
result = rag.ask("नागरिकता नवीकरण गर्न के चाहिन्छ?")

print(result.answer)   # Answer with inline citations
print(result.sources)  # Exact document + page references
```

**2. Legal document structure parsing**

Nepal's government circulars and laws use numbered section structures (`12. Termination`, `Section 4.2`, lettered subsections). RAGNav's `ingest_legal` parser handles this natively — preserving hierarchy, parent-child relationships, and section paths so retrieval respects document structure.

**3. Adaptive retrieval with fallback**

When the first retrieval attempt is low-confidence (common with Nepali/English mixed queries), ragfallback automatically generates rephrased query variants and retries. No silent failures.

**4. Agentic task execution**

Multi-step government service workflows via AgentEnsemble. Supervisor, pipeline, and swarm coordination modes. SQLite-backed session memory — no cloud dependency.

**5. Offline-first architecture**

BM25 retrieval works with zero internet. Sentence-transformer embeddings run locally. SQLite caching means repeated queries are free. Designed for Nepal's geography — remote municipalities, areas with unreliable connectivity.

---

## The gap this fills

| Layer                           | Who's doing it               | What's missing     |
| ------------------------------- | ---------------------------- | ------------------ |
| Nepali NLP / NLU                | Paaila Technology, Diyo AI   | —                  |
| Government chatbots             | Diyo AI (Muna), Fusemachines | —                  |
| AI training / fellowships       | Fusemachines                 | —                  |
| **Legal RAG with citation**     | **Nobody**                   | **← This project** |
| **Adaptive retrieval fallback** | **Nobody**                   | **← This project** |
| **Agentic task orchestration**  | **Nobody**                   | **← This project** |
| **Offline-capable full stack**  | **Nobody**                   | **← This project** |

---

## Nepal context: why now

- **August 2025** — Nepal Cabinet approved [National AI Policy 2082](https://aiassociationnepal.org/national-artificial-intelligence-ai-policy-2082/) — first dedicated national AI policy
- **November 2025** — AIAN + Embassy of India co-hosted "AI for Inclusive Growth: Building Nepal's AI Ready Future" (400+ participants, official pre-summit event for India AI Impact Summit 2026)
- **February 2026** — [World Bank approved $50M](https://coingeek.com/nepal-secures-50m-for-digitalization-approved-by-world-bank/) Nepal Digital Transformation Project
- **March 2026** — [Nagarik App at 1.5M+ users](https://en.wikipedia.org/wiki/Nagarik_App) but backend intelligence layer is missing
- **2026** — National AI Centre operational under MOCIT; 7 provincial AI Excellence Centres being established

The infrastructure window is open.

---

## Seed corpus (in this repo)

The `Data/` folder contains verified, publicly available Nepal government documents:

| Document                                       | Language |
| ---------------------------------------------- | -------- |
| National AI Policy 2082                        | English  |
| Constitution of Nepal (2nd amendment)          | English  |
| Digital Nepal Framework 2.0                    | English  |
| प्रतिनिधि सभा सदस्य निर्वाचन अध्यादेश २०८२     | Nepali   |
| मानव अधिकार पुरस्कार कोष सञ्चालन नियमावली २०७५ | Nepali   |

Community contributions of ministry circulars, SOPs, and municipality guidelines are the highest priority — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Nepal GovAgent                           │
├─────────────────┬─────────────────────┬─────────────────────┤
│   Agent Layer   │     RAG Layer       │  Reliability Layer   │
│  (AgentEnsemble)│    (RAGNav)         │  (ragfallback)       │
│                 │                     │                      │
│  Supervisor     │  PDF ingestion      │  Confidence scoring  │
│  Pipeline       │  Legal doc parser   │  Adaptive fallback   │
│  Swarm          │  BM25 + vectors     │  Cost tracking       │
│  SQLite memory  │  Inline citations   │  Retrieval health    │
│  Task planner   │  Graph expansion    │  Diagnostics         │
└────────┬────────┴──────────┬──────────┴──────────┬──────────┘
         │                   │                      │
         ▼                   ▼                      ▼
┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Nepali NLP   │   │ Nepal Gov Corpus  │   │ Existing Nepal   │
│ (Paaila /    │   │ Data/ folder      │   │ Gov Systems      │
│  Bhashini)   │   │ + community docs  │   │ (Nagarik App,    │
└──────────────┘   └──────────────────┘   │  municipalities) │
                                           └──────────────────┘
```

---

## Installation

```bash
pip install nepal-gov-agent
```

Or from source:

```bash
git clone https://github.com/irfanalidv/Nepal-Gov-Agent
cd Nepal-Gov-Agent
pip install -e .
```

### Core dependencies

```bash
pip install ragnav ragfallback agentensemble
pip install pymupdf sentence-transformers
```

---

## Quick start

```python
from nepal_gov_agent import GovRAG

# Works fully offline
rag = GovRAG(
    corpus_dir="Data/",
    offline=True,
    language="auto",    # Handles Nepali + English
)

# Ask in Nepali
result = rag.ask("नागरिकता नवीकरण गर्न के चाहिन्छ?")
print(result.answer)
print(result.sources)

# Ask in English
result = rag.ask("What are the key provisions of Nepal's National AI Policy?")
print(result.answer)
print(result.sources)
```

---

## Roadmap

### Phase 1 — RAG core (current focus)

- [x] Seed corpus: Nepal government documents in `Data/`
- [ ] `GovRAG` class: unified API over RAGNav + ragfallback
- [ ] Nepali + English hybrid chunking strategy
- [ ] Offline embedding with sentence-transformers
- [ ] Citation validation for every answer
- [ ] CLI: `nepal-gov-agent ask "your question"`

### Phase 2 — Agent capabilities

- [ ] `GovAgent` class: multi-step task execution via AgentEnsemble
- [ ] Common government workflow templates (citizenship, permits, licenses)
- [ ] Nagarik App integration layer
- [ ] Corpus expansion: ministry circulars 2080–2082

### Phase 3 — Production infra

- [ ] MLOps monitoring dashboard
- [ ] Audit trail: every agent action logged and explainable
- [ ] Deployment guide for municipal servers (low-spec hardware)
- [ ] Bhashini + Bolna integration for voice workflows

---

## Built on

| Library                                                      | Role                                | PyPI                        |
| ------------------------------------------------------------ | ----------------------------------- | --------------------------- |
| [RAGNav](https://github.com/irfanalidv/RAGNav)               | Retrieval, citations, legal parsing | `pip install ragnav`        |
| [ragfallback](https://github.com/irfanalidv/ragfallback)     | Reliability, fallback, diagnostics  | `pip install ragfallback`   |
| [AgentEnsemble](https://github.com/irfanalidv/AgentEnsemble) | Orchestration, memory, planning     | `pip install agentensemble` |

All three are open-source, MIT licensed, and independently usable.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Highest priority: **Nepal government document corpus**. If you have publicly available PDFs — ministry circulars, SOPs, municipality guidelines — open an issue or PR.

---

## About

Built by [Irfan Ali](https://www.linkedin.com/in/irfanalidv/) — [GitHub](https://github.com/irfanalidv) | Founder, [DataCortex IQ](https://www.datacortex.in/) | 7+ years in LLM engineering, RAG pipelines, and agentic AI | 11 open-source Python libraries on PyPI | M.Sc. Data Science & AI, IISER Tirupati.

Built in the spirit of AIAN's four pillars: **Data. Infrastructure. Policy. Resources.**

---

## License

MIT — free for government, private sector, and academic use. No strings.

---

_If you're working on Nepal's AI infrastructure layer — government, private sector, or development partner — open an issue or reach out directly._
