import pytest
import chromadb
from unittest.mock import Mock
from hybridix.models import Chunk, ChunkMetadata
from hybridix.indexing import dense_indexer

def test_reset_creates_missing_collection(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path / "dense"))
    monkeypatch.setattr(
        dense_indexer,
        "get_client",
        lambda: client
    )
    collection = dense_indexer.reset_collection()

    assert collection.name == dense_indexer.settings.chroma_collection_name
    assert collection.count() == 0

def test_embedding_failure_preserves_existing_index(monkeypatch):
    chunk = Chunk(
        content="Discord rate limit documentation.",
        metadata=ChunkMetadata(
            source="docs.mdx",
            file_type="mdx",
            chunk_id="test-chunk",
            strategy="heading",
            chunk_index=0,
            heading="Rate Limits"
        )
    )

    reset = Mock()

    monkeypatch.setattr(
        dense_indexer,
        "reset_collection",
        reset
    )

    monkeypatch.setattr(
        dense_indexer,
        "embed_texts",
        Mock(side_effect=RuntimeError("Embedding failed"))
    )

    with pytest.raises(RuntimeError, match="Embedding failed"):
        dense_indexer.build_dense_index([chunk])

    reset.assert_not_called()

def test_successful_indexing_order(monkeypatch):
    chunk = Chunk(
        content="Discord rate limit documentation.",
        metadata=ChunkMetadata(
            source="docs.mdx",
            file_type="mdx",
            chunk_id="test-chunk",
            strategy="heading",
            chunk_index=0,
            heading="Rate Limits"
        )
    )

    steps = []

    embeddings = Mock()
    embeddings.tolist.return_value = [[0.1, 0.2]]

    def fake_embed(texts):
        steps.append("embed")
        return embeddings

    collection = Mock()
    collection.upsert.side_effect = (
        lambda **kwargs: steps.append("upsert")
    )

    def fake_reset():
        steps.append("reset")
        return collection

    monkeypatch.setattr(
        dense_indexer, "embed_texts", fake_embed
    )
    monkeypatch.setattr(
        dense_indexer, "reset_collection", fake_reset
    )

    dense_indexer.build_dense_index([chunk])

    assert steps == ["embed", "reset", "upsert"]
    collection.upsert.assert_called_once()