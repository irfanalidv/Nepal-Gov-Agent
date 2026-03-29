# Contributing to Nepal GovAgent

First off — thank you. This project exists to fill a real gap in Nepal's AI infrastructure layer, and it only works if the community builds it together.

---

## Who should contribute

- Developers in Nepal's tech ecosystem (Kathmandu, Pokhara, anywhere)
- AI/ML engineers interested in low-resource language systems
- Government tech practitioners who know the workflows firsthand
- Anyone who has worked with Nepal's digital public infrastructure

You don't need to be Nepali. You do need to care about building things that actually work in context.

---

## What we need most right now

### 1. Nepal government document corpus
This is the highest priority. We need publicly available government documents to build and test the RAG pipeline:

- Ministry circulars (any ministry)
- Nepal government policy documents
- Legal texts (acts, regulations)
- Municipality service guidelines
- Nagarik App service documentation

If you have access to any of these in PDF or text form and they are publicly available, open an issue or submit a PR adding them to `corpus/raw/`.

### 2. Nepali language test cases
We need question-answer pairs in both Nepali (Devanagari) and English to evaluate retrieval quality:

```
Question: नागरिकता नवीकरण गर्न के चाहिन्छ?
Expected source: Ministry of Home Affairs circular, section X
```

Add test cases to `tests/nepali_qa.json`.

### 3. Hardware testing
Nepal's government infrastructure runs on modest hardware. We need people to test the library on:

- Low-spec servers (4GB RAM, CPU only)
- Common setups in Nepal's municipal offices
- Raspberry Pi or equivalent edge devices

Open an issue with your hardware specs and results.

### 4. Code contributions
See the roadmap in README.md. Current open areas:

- Nepali text chunking strategy
- Devanagari + Romanized Nepali normalization
- Ollama / llama.cpp integration
- Document ingestion pipeline (PDF → chunks)
- Vector store wrapper (local, no cloud)

---

## How to contribute code

### Setup

```bash
git clone https://github.com/irfanalidv/Nepal-Gov-Agent
cd Nepal-Gov-Agent
pip install -e ".[dev]"
```

### Before submitting a PR

- Run existing tests: `pytest tests/`
- Add tests for any new functionality
- Keep it simple — this library runs offline on modest hardware, so avoid heavy dependencies
- Document in English and Nepali where possible

### PR process

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Open a PR with a clear description of what and why
5. Reference any related issues

---

## How to report issues

Open a GitHub issue with:

- What you were trying to do
- What happened
- Your OS, Python version, hardware specs
- Minimal code to reproduce

---

## Principles

**Offline first.** Every feature must work without internet connectivity. Nepal's geography means connectivity cannot be assumed.

**Nepali language native.** Don't treat Nepali as an afterthought. Test in Devanagari, not just English.

**Modest hardware.** If it only runs on a GPU server, it won't reach the municipalities that need it most.

**No proprietary lock-in.** No mandatory cloud APIs. No OpenAI dependency in the core library.

---

## Questions?

Open an issue or reach out directly:

- GitHub: [@irfanalidv](https://github.com/irfanalidv)
- LinkedIn: [Irfan Ali](https://www.linkedin.com/in/irfanalidv/)

---

MIT Licensed. Build freely.
