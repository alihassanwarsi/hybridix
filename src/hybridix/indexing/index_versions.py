import os
import re
import json
import tempfile
from pathlib import Path
from hybridix.config import settings
from hybridix.models import Chunk
from hybridix.indexing.validator import validate_index_pair

def read_active_version(index_root: Path) -> str | None:
    active_file = index_root / "active.json"

    if not active_file.exists():
        return None

    data = json.loads(active_file.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Invalid active index file.")

    version = data.get("version")

    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version):
        raise ValueError("Invalid active index version")

    return version

def resolve_index_paths(index_root: Path, fallback_dense: Path, fallback_sparse: Path) -> tuple[Path, Path]:
    version = read_active_version(index_root)

    if version is None:
        return fallback_dense, fallback_sparse

    version_root = index_root / version
    dense_path = version_root / "dense"
    sparse_path = version_root / "sparse"

    if not (dense_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Active dense index missing: {dense_path}")

    if not (sparse_path / "params.index.json").is_file():
        raise FileNotFoundError(f"Active sparse index missing: {sparse_path}")

    return dense_path, sparse_path

def get_active_index_paths() -> tuple[Path, Path]:
    return resolve_index_paths(
        index_root=settings.index_root,
        fallback_dense=settings.dense_index_path,
        fallback_sparse=settings.sparse_index_path,
    )

def activate_index_version(version: str, chunks: list[Chunk], *, index_root: Path | None = None) -> None:
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version):
        raise ValueError("Invalid index version name.")

    root = index_root if index_root is not None else settings.index_root
    version_root = root / version

    if not version_root.is_dir():
        raise FileNotFoundError(f"Index version not found: {version}")

    dense_path = version_root / "dense"
    sparse_path = version_root / "sparse"

    validate_index_pair(chunks, dense_path, sparse_path)

    active_file = root / "active.json"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".active-",
            suffix=".tmp",
            delete=False
        ) as temp:
            temp_path = Path(temp.name)
            json.dump({"version": version}, temp)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())

        os.replace(temp_path, active_file)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)