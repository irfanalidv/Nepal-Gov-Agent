from nepal_gov_agent.preprocess import preprocess_query


def test_preprocess_query_nfc_and_suffix():
    q = preprocess_query("नेपालको राष्ट्रिय AI नीतिको उद्देश्य के हो?")
    assert "के हो" not in q
    assert "नेपालको" in q


def test_preprocess_query_collapses_whitespace():
    assert preprocess_query("hello   world") == "hello world"
