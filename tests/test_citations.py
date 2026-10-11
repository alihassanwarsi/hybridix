import pytest
from hybridix.generation.citations import resolve_citations
from hybridix.models import Chunk, ChunkMetadata, RetrievalResult

def make_result(chunk_id: str, source: str) -> RetrievalResult:
    chunk = Chunk(
        content=f"Evidence from {source}",
        metadata=ChunkMetadata(
            source=source,
            file_type="mdx",
            chunk_id=chunk_id,
            strategy="heading",
            chunk_index=0
        )
    )

    return RetrievalResult(chunk=chunk, score=1.0)

def test_citations_resolve_to_correct_chunks():
    results = [
        make_result("chunk-a", "first.mdx"),
        make_result("chunk-b", "second.mdx"),
    ]

    citations = resolve_citations("Answer supported by [2] and [1].", results)

    assert len(citations) == 2

    assert citations[0].number == 2
    assert citations[0].chunk_id == "chunk-b"
    assert citations[0].source == "second.mdx"
    assert citations[0].content == "Evidence from second.mdx"

    assert citations[1].chunk_id == "chunk-a"


def test_repeated_citations_are_deduplicated():
    results = [make_result("chunk-a", "first.mdx")]

    citations = resolve_citations("First claim [1]. Second claim [1].", results)

    assert len(citations) == 1

def test_unknown_citation_is_rejected():
    results = [make_result("chunk-a", "first.mdx")]

    with pytest.raises(ValueError, match="Unknown citation"):
        resolve_citations("Invented citation [99].", results)

def test_answer_without_citations():
    results = [make_result("chunk-a", "first.mdx")]

    citations = resolve_citations("I do not have enough evidence to answer.", results)

    assert citations == []

def test_missing_required_citation_is_rejected():
    results = [make_result("chunk-a", "first.mdx")]

    with pytest.raises(ValueError, match="missing citations"):
        resolve_citations("Guild commands update instantly.", results, require_citations=True)