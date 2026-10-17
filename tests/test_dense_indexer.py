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

def test_get_client_uses_explicit_paths(tmp_path, monkeypatch):
    calls = []

    def fake_client(path):
        calls.append(path)
        return object()

    monkeypatch.setattr(
        dense_indexer.chromadb,
        "PersistentClient",
        fake_client
    )

    dense_indexer._cached_client.cache_clear()

    first_path = tmp_path / "version_001" / "dense"
    second_path = tmp_path / "version_002" / "dense"

    first = dense_indexer.get_client(first_path)
    second = dense_indexer.get_client(second_path)
    first_again = dense_indexer.get_client(first_path)

    assert first is first_again
    assert first is not second
    assert calls == [
        str(first_path.resolve()),
        str(second_path.resolve())
    ]

    dense_indexer._cached_client.cache_clear()


def test_get_collection_uses_selected_path(tmp_path, monkeypatch):
    selected_path = tmp_path / "version_002" / "dense"

    fake_client = object()
    fake_collection = object()

    def fake_get_client(index_path):
        assert index_path == selected_path
        return fake_client

    monkeypatch.setattr(
        dense_indexer, "get_client", fake_get_client
    )

    monkeypatch.setattr(
        dense_indexer, "_get_or_create_collection",
        lambda client: fake_collection
    )

    result = dense_indexer.get_collection(selected_path)

    assert result is fake_collection