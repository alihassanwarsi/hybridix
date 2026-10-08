import pytest
from hybridix.retrieval.hybrid_retriever import rrf_score

def test_rrf_score_decreases_with_rank():
    assert rrf_score(1) > rrf_score(2)

def test_rrf_score_uses_expected_formula():
    assert rrf_score(1) == pytest.approx(1/61)

def test_rrf_rejects_invalid_rank():
    with pytest.raises(ValueError):
        rrf_score(0)