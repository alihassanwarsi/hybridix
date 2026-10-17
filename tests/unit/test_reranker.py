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


def test_reranked_search_uses_active_paths(monkeypatch, tmp_path):
    dense_path = tmp_path / "version_002" / "dense"
    sparse_path = tmp_path / "version_002" / "sparse"

    resolver = Mock(return_value=(dense_path, sparse_path))
    dense = Mock(return_value=[])
    sparse = Mock(return_value=[])

    monkeypatch.setattr(
        reranker, "get_active_index_paths", resolver
    )
    monkeypatch.setattr(reranker, "dense_search", dense)
    monkeypatch.setattr(reranker, "sparse_search", sparse)

    results = reranker.reranked_search(
        "test question", top_k=3, candidate_k=5
    )

    assert results == []
    resolver.assert_called_once_with()

    dense.assert_called_once_with(
        "test question",
        top_k=5,
        index_path=dense_path
    )

    sparse.assert_called_once_with(
        "test question",
        top_k=5,
        index_path=sparse_path
    )