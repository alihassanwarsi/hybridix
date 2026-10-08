from hybridix.evaluation.retrieval import hit_at_k, reciprocal_rank
from hybridix.models import Chunk, ChunkMetadata, RetrievalResult

def make_result(source: str, heading: str | None = None) -> RetrievalResult:
    chunk = Chunk(
        content="test",
        metadata=ChunkMetadata(
            chunk_id="test-id",
            source=source,
            file_type="mdx",
            title=None,
            strategy="heading",
            chunk_index=0,
            heading=heading
        )
    )

    return RetrievalResult(chunk=chunk, score=1.0)

def test_hit_at_k_finds_relevant_result():
    results = [
        make_result("wrong.mdx"),
        make_result("correct.mdx"),
    ]

    relevant = [
        {"source": "correct.mdx", "heading": None}
    ]

    assert hit_at_k(results, relevant, k=2) == 1.0

def test_hit_at_k_returns_zero_when_missing():
    results = [
        make_result("wrong.mdx")
    ]

    relevant = [
        {"source": "correct.mdx", "heading": None}
    ]

    assert hit_at_k(results, relevant, k=2) == 0.0

def test_reciprocal_rank_uses_result_position():
    results = [
        make_result("wrong.mdx"),
        make_result("correct.mdx"),
    ]

    relevant = [
        {"source": "correct.mdx", "heading": None}
    ]

    assert reciprocal_rank(results, relevant, k=2) == 0.5