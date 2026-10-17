import numpy as np
import pytest
from unittest.mock import Mock
from hybridix.config import settings
from hybridix.models import Chunk, ChunkMetadata
from hybridix.indexing import dense_indexer
from hybridix.indexing.versioning import builder as version_builder
from hybridix.indexing.versioning.active import activate_index_version, get_active_index_paths, read_active_version
from hybridix.retrieval import dense_retriever
from hybridix.retrieval.sparse_retriever import sparse_search

def make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            source="test.mdx",
            file_type="mdx",
            chunk_id=chunk_id,
            strategy="heading",
            chunk_index=0
        )
    )

def test_safe_index_replacement(tmp_path, monkeypatch):
    root = tmp_path / "indexes"

    monkeypatch.setattr(settings, "index_root", root)

    monkeypatch.setattr(
        settings, "dense_index_path", tmp_path / "unused_dense"
    )
    monkeypatch.setattr(
        settings, "sparse_index_path", tmp_path / "unused_sparse"
    )

    monkeypatch.setattr(
        dense_indexer,
        "embed_texts",
        lambda texts: np.tile(
            np.array([[1.0, 0.0]]), (len(texts), 1)
        )
    )

    monkeypatch.setattr(
        dense_retriever,
        "embed_query",
        lambda query: np.array([[1.0, 0.0]])
    )

    old_chunk = make_chunk("old-1", "Old documentation about limits.")
    new_chunk = make_chunk("new-1", "New documentation about gateway.")

    version_builder.build_index_version(
        [old_chunk], "version_001", index_root=root
    )

    activate_index_version(
        "version_001", [old_chunk], index_root=root
    )

    old_dense, old_sparse = get_active_index_paths()

    assert read_active_version(root) == "version_001"

    version_builder.build_index_version(
        [new_chunk], "version_002", index_root=root
    )

    assert read_active_version(root) == "version_001"

    activate_index_version(
        "version_002", [new_chunk], index_root=root
    )

    new_dense, new_sparse = get_active_index_paths()

    assert read_active_version(root) == "version_002"
    assert new_dense == root / "version_002" / "dense"
    assert new_sparse == root / "version_002" / "sparse"

    dense_results = dense_retriever.dense_search(
        "gateway", top_k=1, index_path=new_dense
    )

    sparse_results = sparse_search(
        "gateway", top_k=1, index_path=new_sparse
    )

    assert dense_results[0].chunk.metadata.chunk_id == "new-1"
    assert sparse_results[0].chunk.metadata.chunk_id == "new-1"

    assert (old_dense / "chroma.sqlite3").exists()
    assert (old_sparse / "params.index.json").exists()

    monkeypatch.setattr(
        version_builder,
        "build_sparse_index",
        Mock(side_effect=RuntimeError("BM25 failed"))
    )

    with pytest.raises(RuntimeError, match="BM25 failed"):
        version_builder.build_index_version(
            [make_chunk("broken-1", "Broken build.")],
            "version_003",
            index_root=root
        )

    assert read_active_version(root) == "version_002"
    assert get_active_index_paths() == (new_dense, new_sparse)

def test_failed_pointer_replacement_preserves_active_version(tmp_path, monkeypatch):
    from hybridix.indexing.versioning import active as index_versions

    root = tmp_path / "indexes"
    (root / "version_002").mkdir(parents=True)

    active_file = root / "active.json"
    active_file.write_text(
        '{"version": "version_001"}',
        encoding="utf-8"
    )

    monkeypatch.setattr(
        index_versions,
        "validate_index_pair",
        Mock()
    )

    monkeypatch.setattr(
        index_versions.os,
        "replace",
        Mock(side_effect=OSError("Simulated disk error"))
    )

    with pytest.raises(OSError, match="Simulated disk error"):
        index_versions.activate_index_version(
            "version_002",
            [object()],
            index_root=root
        )

    assert read_active_version(root) == "version_001"
    assert list(root.glob(".active-*.tmp")) == []