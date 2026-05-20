# Seed corpus (not stored in git)

PDFs are **not** committed to this repository (size and redistribution). Fetch the five seed documents with:

```python
from nepal_gov_agent import download_corpus

corpus_dir = download_corpus()  # → ./nepal_gov_data/
```

Or from the CLI after `pip install nepal-gov-agent`:

```bash
python -c "from nepal_gov_agent import download_corpus; download_corpus()"
```

Place your own Nepal government PDFs in any folder and pass that path to `GovRAG(corpus_dir=...)`.
