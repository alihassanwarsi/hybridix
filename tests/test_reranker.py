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