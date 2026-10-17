import json
import pytest
from pathlib import Path
from hybridix.indexing.index_versions import read_active_version, resolve_index_paths

def test_no_active_version(tmp_path):
    result = read_active_version(tmp_path)

    assert result is None

def test_valid_active_version(tmp_path):
    active_file = tmp_path / "active.json"

    active_file.write_text(
        json.dumps({"version": "version_002"}),
        encoding="utf-8"
    )

    result = read_active_version(tmp_path)

    assert result == "version_002"

def test_invalid_active_version(tmp_path):
    active_file = tmp_path / "active.json"

    active_file.write_text(
        json.dumps({"version": "../old_index"}),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        read_active_version(tmp_path)

def test_resolve_paths_without_active_version(tmp_path):
    dense = Path("existing/dense")
    sparse = Path("existing/sparse")

    result = resolve_index_paths(tmp_path, dense, sparse)

    assert result == (dense, sparse)

def test_resolve_paths_with_active_version(tmp_path):
    (tmp_path / "active.json").write_text(
        '{"version": "version_002"}',
        encoding="utf-8"
    )

    result = resolve_index_paths(
        tmp_path,
        Path("existing/dense"),
        Path("existing/sparse")
    )

    assert result == (
        tmp_path / "version_002" / "dense",
        tmp_path / "version_002" / "sparse"
    )