from nepal_gov_agent.ingest import ingest_corpus


def test_ingest_corpus_loads_pdfs():
    docs, blocks = ingest_corpus("Data/", verbose=False)
    assert len(docs) >= 1
    assert len(blocks) >= 10


def test_ingest_corpus_has_nepali_docs():
    docs, blocks = ingest_corpus("Data/", verbose=False)
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) > 0
