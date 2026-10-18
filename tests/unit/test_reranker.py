from unittest.mock import Mock
from hybridix.models import Chunk, ChunkMetadata, RetrievalResult
from hybridix.retrieval import reranker

def make_result(chunk_id: str, content: str) -> RetrievalResult:
    chunk = Chunk(
        content=content,
        metadata=ChunkMetadata(
            source="test.mdx",
            file_type="mdx",
            chunk_id=chunk_id,
            strategy="heading",
            chunk_index=0
        )
    )

    return RetrievalResult(chunk=chunk, score=0.0)

def test_rerank_returns_empty_list():
    assert reranker.rerank("test query", []) == []

def test_rerank_orders_by_model_score(monkeypatch):
    results = [
        make_result("a", "less relevant"),
        make_result("b", "most relevant")
    ]

    class FakeModel:
        def predict(self, pairs):
            return [0.2, 0.9]

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda: FakeModel()
    )

    ranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=2
    )

    assert ranked[0].chunk.metadata.chunk_id == "b"
    assert ranked[1].chunk.metadata.chunk_id == "a"

def test_reranked_search_uses_rrf_candidates(monkeypatch):
    candidates = [
        make_result("a", "first"),
        make_result("b", "second"),
    ]

    hybrid = Mock(return_value=candidates)
    rerank_mock = Mock(return_value=[candidates[1]])

    monkeypatch.setattr(reranker, "hybrid_search", hybrid)
    monkeypatch.setattr(reranker, "rerank", rerank_mock)

    results = reranker.reranked_search(
        "test question",
        top_k=1,
        candidate_k=30
    )

    assert results == [candidates[1]]

    hybrid.assert_called_once_with(
        "test question",
        top_k=30,
        candidate_k=30
    )

    rerank_mock.assert_called_once_with(
        query="test question",
        results=candidates,
        top_k=1
    )