import pytest
import numpy as np
from hybridix.models import Chunk, ChunkMetadata
from hybridix.indexing import dense_indexer, sparse_indexer
from hybridix.indexing.versioning.validator import validate_index_pair

def test_validate_index_pair(tmp_path, monkeypatch):
    dense_path = tmp_path / "new_version" / "dense"
    sparse_path = tmp_path / "new_version" / "sparse"

    chunk = Chunk(
        content="Discord uses rate limit buckets.",
        metadata=ChunkMetadata(
            source="docs.mdx",
            file_type="mdx",
            chunk_id="chunk-1",
            strategy="heading",
            chunk_index=0
        )
    )

    monkeypatch.setattr(
        dense_indexer,
        "embed_texts",
        lambda texts: np.array([[0.1, 0.2]])
    )

    dense_indexer.build_dense_index([chunk], index_path=dense_path)
    sparse_indexer.build_sparse_index([chunk], index_path=sparse_path)

    validate_index_pair([chunk], dense_path, sparse_path)

    missing_chunk = Chunk(
        content="Another document.",
        metadata=ChunkMetadata(
            source="other.mdx",
            file_type="mdx",
            chunk_id="chunk-2",
            strategy="heading",
            chunk_index=1
        )
    )

    with pytest.raises(ValueError, match="chunk IDs"):
        validate_index_pair(
            [chunk, missing_chunk],
            dense_path,
            sparse_path
        )