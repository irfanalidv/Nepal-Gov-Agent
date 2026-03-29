# Nepal GovAgent

> Open-source agentic AI framework for Nepal's government service layer — RAG, multi-step task automation, and MLOps infra designed for offline deployment and Nepali language workflows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
#[![PyPI version](https://img.shields.io/pypi/v/nepal-gov-agent.svg)](https://pypi.org/project/nepal-gov-agent/)
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()

---

## Why this exists

Nepal has a digital government layer. The [Nagarik App](https://nagarikapp.gov.np/) has 1.5M+ registered citizens. Diyo AI has deployed Nepali-language chatbots for Lalitpur and Butwal municipalities. Paaila Technology has built Nepali NLU and TTS. Fusemachines is training hundreds of AI fellows.

**The conversational layer exists. What doesn't exist is the intelligence layer above it.**

Today, Nepal's government AI systems can answer questions. They cannot:

- Complete multi-step tasks on behalf of a citizen ("apply for this permit end-to-end")
- Retrieve and cite specific clauses from Nepal's legal and policy corpus
- Run reliably without internet connectivity in areas with poor network coverage
- Plug into existing systems without cloud API dependency (no data leaving Nepal)
- Monitor and audit themselves at production scale

Nepal's [National AI Policy 2082](https://aiassociationnepal.gov.np) explicitly identifies **infrastructure, data, and sovereignty** as its foundational pillars. The [AI Association of Nepal (AIAN)](https://aiassociationnepal.org) has framed national AI readiness around exactly these four pillars: Data, Infrastructure, Policy, Resources.

This project is the infrastructure answer.

---

## What Nepal GovAgent does

Nepal GovAgent is a framework for building **agentic AI systems** on top of Nepal's existing government digital infrastructure — designed from the ground up for the Nepal context.

### Core capabilities

**1. Agentic task execution**
Multi-step workflow automation for government service tasks. Not just Q&A — actual task completion across multiple systems. A citizen asks to renew a document; the agent retrieves the right form, checks eligibility, pre-fills available data, and guides through submission.

**2. RAG over Nepal's policy and legal corpus**
Retrieval-Augmented Generation over Nepal government circulars, policy documents, laws, and ministry guidelines — with proper source citation. Every answer traceable to a real document.

**3. Offline-first architecture**
Runs on modest hardware without internet connectivity. Designed for Nepal's geography — remote municipalities, areas with unreliable connectivity. No mandatory cloud API calls.

**4. Nepali language native**
Built to work with existing Nepali NLP layers (Devanagari + Romanized Nepali). Integrates with Bhashini (India's open language AI) for cross-border India-Nepal AI corridor use cases.

**5. MLOps and auditability**
Production monitoring, logging, and audit trails built in. Government deployments need accountability — every agent action is logged and explainable.

---

## The gap this fills

| Layer | Who's doing it | What's missing |
|---|---|---|
| Nepali NLP / NLU | Paaila Technology, Diyo AI | — |
| Government chatbots | Diyo AI (Muna), Fusemachines | — |
| AI training / fellowships | Fusemachines | — |
| **Agentic orchestration** | **Nobody** | **← This project** |
| **RAG over policy corpus** | **Nobody** | **← This project** |
| **Offline-capable infra** | **Nobody** | **← This project** |
| **MLOps / audit layer** | **Nobody** | **← This project** |

---

## Context: Nepal's AI moment

- **August 2025** — Nepal Cabinet approved [National AI Policy 2082](https://aiassociationnepal.org/national-artificial-intelligence-ai-policy-2082/) — its first dedicated national AI policy
- **November 2025** — AIAN and Embassy of India co-hosted "AI for Inclusive Growth: Building Nepal's AI Ready Future" (400+ participants, senior ministers, official pre-summit event for India AI Impact Summit 2026)
- **February 2026** — [World Bank approved $50M](https://coingeek.com/nepal-secures-50m-for-digitalization-approved-by-world-bank/) for Nepal Digital Transformation Project
- **2026** — Nepal AI Conference 2026 underway; National AI Centre operational under MOCIT
- **March 2026** — [Nagarik App at 1.5M+ users](https://en.wikipedia.org/wiki/Nagarik_App) but institutional adoption still faces backend intelligence gaps

The infrastructure window is open. This project aims to fill it before it closes.

---

## Architecture (planned)

```
┌─────────────────────────────────────────────────────┐
│                  Nepal GovAgent                      │
├──────────────┬──────────────────┬────────────────────┤
│  Agent Core  │   RAG Pipeline   │   MLOps Layer      │
│              │                  │                    │
│  - Planner   │  - Doc ingestion │  - Logging         │
│  - Executor  │  - Nepali chunker│  - Monitoring      │
│  - Memory    │  - Vector store  │  - Audit trails    │
│  - Tools     │  - Retriever     │  - Dashboards      │
└──────┬───────┴────────┬─────────┴──────────┬─────────┘
       │                │                    │
       ▼                ▼                    ▼
┌─────────────┐  ┌─────────────┐   ┌────────────────┐
│ Nepali NLP  │  │ Gov Doc     │   │ Existing Nepal │
│ Layer       │  │ Corpus      │   │ Gov Systems    │
│ (Paaila /   │  │ (circulars, │   │ (Nagarik App,  │
│  Bhashini)  │  │  laws, SOP) │   │  municipality  │
└─────────────┘  └─────────────┘   │  portals)      │
                                   └────────────────┘
```

---

## Roadmap

### Phase 1 — Foundation (Current)
- [ ] Project scaffolding and architecture
- [ ] Nepal government document ingestion pipeline
- [ ] Basic RAG over policy corpus (Nepali + English)
- [ ] Offline LLM integration (llama.cpp / Ollama)
- [ ] Core agent loop with tool calling

### Phase 2 — Agent Capabilities
- [ ] Multi-step task planner for common government workflows
- [ ] Nagarik App API integration layer
- [ ] Nepali language chunking and retrieval optimization
- [ ] Citation and source tracing for every RAG response

### Phase 3 — Production Infra
- [ ] MLOps monitoring and logging
- [ ] Audit trail system
- [ ] Deployment guide for municipal servers
- [ ] Bhashini integration for India-Nepal corridor use cases

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

> **Note:** Active development. APIs will change. Star and watch for updates.

---

## Quick start

```python
from nepal_gov_agent import GovAgent

agent = GovAgent(
    language="nepali",
    offline_mode=True,        # No external API calls
    corpus="gov_docs/",       # Your Nepal gov document folder
)

response = agent.run(
    "नागरिकता नवीकरण गर्न के चाहिन्छ?",  # "What is needed to renew citizenship?"
)

print(response.answer)
print(response.sources)      # Cited government documents
```

---

## Contributing

Contributions welcome — especially from Nepal's developer community.

Priority contributions needed:
- Nepal government document corpus (circulars, policy PDFs, ministry guidelines)
- Nepali language test cases
- Integration with existing municipality systems
- Hardware testing on low-spec servers common in Nepal's government infrastructure

See CONTRIBUTING.md for guidelines (coming soon).
---

## About
Built by [Irfan Ali](https://www.linkedin.com/in/irfanalidv/) — [GitHub](https://github.com/irfanalidv) | Founder, [DataCortex IQ](https://www.datacortex.in/) | 7+ years in LLM engineering, RAG pipelines, and agentic AI | 11 open-source Python libraries on PyPI | M.Sc. Data Science & AI, IISER Tirupati.

Built in the spirit of AIAN's four pillars: **Data. Infrastructure. Policy. Resources.**

---

## License

MIT — free for government, private sector, and academic use. No strings.

---

*If you're working on Nepal's AI infrastructure layer — from the government side, private sector, or development partner side — open an issue or reach out directly.*
