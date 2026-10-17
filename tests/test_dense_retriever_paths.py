from types import SimpleNamespace
from hybridix.retrieval import dense_retriever

def test_dense_search_uses_selected_path(tmp_path, monkeypatch):
    selected_path = tmp_path / "version_002" / "dense"

    def fake_get_collection(index_path):
        assert index_path == selected_path

        return SimpleNamespace(
            query=lambda **kwargs: {
                "ids": [["chunk-1"]],
                "documents": [["Test document"]],
                "metadatas": [[{
                    "source": "test.mdx",
                    "file_type": "mdx",
                    "strategy": "heading",
                    "chunk_index": 0
                }]],
                "distances": [[0.1]]
            }
        )

    monkeypatch.setattr(
        dense_retriever,
        "get_collection",
        fake_get_collection
    )

    monkeypatch.setattr(
        dense_retriever,
        "embed_query",
        lambda query: SimpleNamespace(
            tolist=lambda: [[0.1, 0.2]]
        )
    )

    results = dense_retriever.dense_search(
        "test question",
        top_k=1,
        index_path=selected_path
    )

    assert len(results) == 1
    assert results[0].chunk.metadata.chunk_id == "chunk-1"