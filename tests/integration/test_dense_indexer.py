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
    monkeypatch.setattr(dense_indexer, "reset_collection", reset)
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
    collection.upsert.side_effect = lambda **kwargs: steps.append("upsert")

    def fake_reset():
        steps.append("reset")
        return collection

    monkeypatch.setattr(dense_indexer, "embed_texts", fake_embed)
    monkeypatch.setattr(dense_indexer, "reset_collection", fake_reset)

    dense_indexer.build_dense_index([chunk])

    assert steps == ["embed", "reset", "upsert"]
    collection.upsert.assert_called_once()

def test_get_client_uses_explicit_paths(tmp_path, monkeypatch):
    calls = []

    def fake_client(path):
        calls.append(path)
        return object()

    monkeypatch.setattr(dense_indexer.chromadb, "PersistentClient", fake_client)
    dense_indexer._cached_client.cache_clear()

    first_path = tmp_path / "version_001" / "dense"
    second_path = tmp_path / "version_002" / "dense"

    first = dense_indexer.get_client(first_path)
    second = dense_indexer.get_client(second_path)
    first_again = dense_indexer.get_client(first_path)

    assert first is first_again
    assert first is not second
    assert calls == [str(first_path.resolve()), str(second_path.resolve())]

    dense_indexer._cached_client.cache_clear()

def test_get_collection_uses_selected_path(tmp_path, monkeypatch):
    selected_path = tmp_path / "version_002" / "dense"
    selected_path.mkdir(parents=True)
    (selected_path / "chroma.sqlite3").touch()

    client = Mock()
    collection = Mock()
    client.get_collection.return_value = collection

    get_client = Mock(return_value=client)
    monkeypatch.setattr(dense_indexer, "get_client", get_client)

    assert dense_indexer.get_collection(selected_path) is collection

    get_client.assert_called_once_with(selected_path)
    client.get_collection.assert_called_once_with(
        name=dense_indexer.settings.chroma_collection_name,
        embedding_function=None,
    )

def test_missing_collection_is_not_created(tmp_path, monkeypatch):
    path = tmp_path / "dense"
    client = chromadb.PersistentClient(path=str(path))
    monkeypatch.setattr(dense_indexer, "get_client", lambda index_path: client)

    with pytest.raises(FileNotFoundError, match="Dense collection"):
        dense_indexer.get_collection(path)

    assert client.list_collections() == []

def test_missing_dense_index_is_not_created(tmp_path, monkeypatch):
    path = tmp_path / "missing_dense"
    get_client = Mock()
    monkeypatch.setattr(dense_indexer, "get_client", get_client)

    with pytest.raises(FileNotFoundError, match="Dense index not found"):
        dense_indexer.get_collection(path)

    assert not path.exists()
    get_client.assert_not_called()


def test_dense_builder_uses_isolated_path(tmp_path, monkeypatch):
    new_path = tmp_path / "new" / "dense"

    chunk = Chunk(
        content="Discord rate limit documentation.",
        metadata=ChunkMetadata(
            source="docs.mdx",
            file_type="mdx",
            chunk_id="test-chunk",
            strategy="heading",
            chunk_index=0
        )
    )

    embeddings = Mock()
    embeddings.tolist.return_value = [[0.1, 0.2]]

    collection = Mock()
    client = Mock()
    get_client = Mock(return_value=client)
    create_collection = Mock(return_value=collection)
    reset = Mock()

    monkeypatch.setattr(dense_indexer, "embed_texts", Mock(return_value=embeddings))
    monkeypatch.setattr(dense_indexer, "get_client", get_client)
    monkeypatch.setattr(dense_indexer, "_get_or_create_collection", create_collection)
    monkeypatch.setattr(dense_indexer, "reset_collection", reset)

    dense_indexer.build_dense_index([chunk], index_path=new_path)

    get_client.assert_called_once_with(new_path)
    create_collection.assert_called_once_with(client)
    collection.upsert.assert_called_once()
    reset.assert_not_called()
