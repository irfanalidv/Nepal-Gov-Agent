from nepal_gov_agent.ingest import ingest_corpus


def test_ingest_corpus_loads_pdfs(corpus_dir):
    docs, blocks = ingest_corpus(corpus_dir)
    assert len(docs) >= 1
    assert len(blocks) >= 3


def test_ingest_corpus_has_nepali_docs(corpus_dir):
    docs, _blocks = ingest_corpus(corpus_dir)
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) > 0
