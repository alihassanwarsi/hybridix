from unittest.mock import Mock
from hybridix.retrieval import hybrid_retriever

def test_hybrid_search_uses_active_index_paths(tmp_path, monkeypatch):
    dense_path = tmp_path / "version_002" / "dense"
    sparse_path = tmp_path / "version_002" / "sparse"

    resolver = Mock(return_value=(dense_path, sparse_path))
    dense = Mock(return_value=[])
    sparse = Mock(return_value=[])

    monkeypatch.setattr(
        hybrid_retriever, "get_active_index_paths", resolver
    )
    monkeypatch.setattr(
        hybrid_retriever, "dense_search", dense
    )
    monkeypatch.setattr(
        hybrid_retriever, "sparse_search", sparse
    )

    results = hybrid_retriever.hybrid_search("rate limits")

    assert results == []

    resolver.assert_called_once_with()
    dense.assert_called_once_with(
        "rate limits", index_path=dense_path
    )
    sparse.assert_called_once_with(
        "rate limits", index_path=sparse_path
    )