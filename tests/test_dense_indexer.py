import chromadb
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