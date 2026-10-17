import re
import json
from pathlib import Path
from hybridix.config import settings
from hybridix.models import Chunk
from hybridix.indexing.dense_indexer import build_dense_index
from hybridix.indexing.sparse_indexer import build_sparse_index
from hybridix.indexing.versioning.validator import validate_index_pair

def build_index_version(chunks: list[Chunk], version: str, *, index_root: Path | None = None,) -> Path:
    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version):
        raise ValueError("Invalid index version name.")

    root = index_root if index_root is not None else settings.index_root
    version_root = root / version

    version_root.mkdir(parents=True, exist_ok=False)

    dense_path = version_root / "dense"
    sparse_path = version_root / "sparse"

    build_dense_index(chunks, index_path=dense_path)
    build_sparse_index(chunks, index_path=sparse_path)

    validate_index_pair(chunks, dense_path, sparse_path)

    manifest = {
    "embedding_model": settings.embedding_model,
    "chunk_count": len(chunks)
    }

    (version_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return version_root