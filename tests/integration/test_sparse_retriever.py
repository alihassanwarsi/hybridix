import pytest
from hybridix.config import settings
from hybridix.indexing.sparse_indexer import build_sparse_index
from hybridix.models import Chunk, ChunkMetadata
from hybridix.retrieval.sparse_retriever import sparse_search

@pytest.mark.parametrize("chunk_count", [1, 2, 29, 30])
def test_sparse_search_handles_small_corpus(tmp_path, monkeypatch, chunk_count):
    monkeypatch.setattr(
        settings,
        "sparse_index_path",
        tmp_path / "sparse"
    )

    chunks = [
        Chunk(
            content=f"Discord gateway sharding document {i}.",
            metadata=ChunkMetadata(
                source="gateway.mdx",
                file_type="mdx",
                chunk_id=f"chunk-{i}",
                strategy="heading",
                chunk_index=i
            )
        )
        for i in range(chunk_count)
    ]

    build_sparse_index(chunks)

    results = sparse_search("gateway sharding", top_k=30)

    assert len(results) == chunk_count

    monkeypatch.setattr(
    settings, "sparse_index_path", tmp_path / "missing"
)

    explicit_results = sparse_search(
        "gateway sharding",
        top_k=30,
        index_path=tmp_path / "sparse"
    )

    assert len(explicit_results) == chunk_count

def test_sparse_index_rejects_empty_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(
        settings,
        "sparse_index_path",
        tmp_path / "sparse"
    )

    with pytest.raises(ValueError, match="No chunks provided"):
        build_sparse_index([])

def test_sparse_search_reports_missing_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        settings,
        "sparse_index_path",
        tmp_path / "missing_index",
    )

    with pytest.raises(FileNotFoundError, match="Sparse index not found"):
        sparse_search("gateway sharding", top_k=5)


def test_sparse_builder_uses_isolated_path(tmp_path, monkeypatch):
    old_path = tmp_path / "old_index"
    new_path = tmp_path / "new_index"

    monkeypatch.setattr(
        settings, "sparse_index_path", old_path
    )

    chunk = Chunk(
        content="Discord uses rate limit buckets.",
        metadata=ChunkMetadata(
            source="docs.mdx",
            file_type="mdx",
            chunk_id="test-1",
            strategy="heading",
            chunk_index=0
        )
    )

    build_sparse_index([chunk], index_path=new_path)

    assert (new_path / "params.index.json").exists()
    assert not old_path.exists()

    with pytest.raises(FileExistsError):
        build_sparse_index([chunk], index_path=new_path)