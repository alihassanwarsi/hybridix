import re
import json
from pathlib import Path
from hybridix.config import settings

def read_active_version(index_root: Path) -> str | None:
    active_file = index_root / "active.json"

    if not active_file.exists():
        return None

    data = json.loads(active_file.read_text(encoding="utf-8"))
    version = data.get("version")

    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version):
        raise ValueError("Invalid active index version")

    return version

def resolve_index_paths(index_root: Path, fallback_dense: Path, fallback_sparse: Path) -> tuple[Path, Path]:
    version = read_active_version(index_root)

    if version is None:
        return fallback_dense, fallback_sparse

    version_root = index_root / version

    return version_root / "dense", version_root / "sparse"

def get_active_index_paths() -> tuple[Path, Path]:
    return resolve_index_paths(
        index_root=settings.index_root,
        fallback_dense=settings.dense_index_path,
        fallback_sparse=settings.sparse_index_path,
    )