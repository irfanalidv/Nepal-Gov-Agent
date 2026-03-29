from nepal_gov_agent import run_benchmark


def test_benchmark_runs(rag):
    results = run_benchmark(rag, verbose=False)
    assert results.total_queries > 0
    assert 0.0 <= results.recall_at_3 <= 1.0
    assert 0.0 <= results.keyword_hit_rate <= 1.0
