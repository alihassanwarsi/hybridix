import pytest
from hybridix.retrieval.hybrid_retriever import rrf_score

def test_rrf_score_decreases_with_rank():
    assert rrf_score(1) > rrf_score(2)

def test_rrf_score_uses_expected_formula():
    assert rrf_score(1) == pytest.approx(1/61)

def test_rrf_rejects_invalid_rank():
    with pytest.raises(ValueError):
        rrf_score(0)

def test_hybrid_search_uses_30_candidates_per_retriever(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from hybridix.models import Chunk, ChunkMetadata, RetrievalResult
    from hybridix.retrieval import hybrid_retriever

    def make_result(chunk_id: str) -> RetrievalResult:
        return RetrievalResult(
            chunk=Chunk(
                content=chunk_id,
                metadata=ChunkMetadata(
                    source="test.mdx",
                    file_type="mdx",
                    chunk_id=chunk_id,
                    strategy="heading",
                    chunk_index=0
                )
            ),
            score=1.0
        )

    dense_path = tmp_path / "new_version" / "dense"
    sparse_path = tmp_path / "new_version" / "sparse"

    resolver = Mock(return_value=(dense_path, sparse_path))
    dense = Mock(return_value=[make_result("a"), make_result("b")])
    sparse = Mock(return_value=[make_result("b"), make_result("c")])

    monkeypatch.setattr(
        hybrid_retriever, "get_active_index_paths", resolver
    )
    monkeypatch.setattr(hybrid_retriever, "dense_search", dense)
    monkeypatch.setattr(hybrid_retriever, "sparse_search", sparse)

    results = hybrid_retriever.hybrid_search(
        "test question",
        top_k=2,
        candidate_k=30
    )

    assert [
        result.chunk.metadata.chunk_id
        for result in results
    ] == ["b", "a"]

    resolver.assert_called_once_with()

    dense.assert_called_once_with(
        "test question",
        top_k=30,
        index_path=dense_path
    )

    sparse.assert_called_once_with(
        "test question",
        top_k=30,
        index_path=sparse_path
    )
