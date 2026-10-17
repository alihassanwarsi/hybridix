import json
import pytest
from pathlib import Path
from unittest.mock import Mock
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
    version_root = tmp_path / "version_002"

    dense_path = version_root / "dense"
    sparse_path = version_root / "sparse"

    dense_path.mkdir(parents=True)
    sparse_path.mkdir(parents=True)

    (dense_path / "chroma.sqlite3").touch()
    (sparse_path / "params.index.json").touch()

    (tmp_path / "active.json").write_text(
        '{"version": "version_002"}',
        encoding="utf-8"
    )

    result = resolve_index_paths(
        tmp_path,
        Path("existing/dense"),
        Path("existing/sparse")
    )

    assert result == (dense_path, sparse_path)

def test_activation_switches_active_version(tmp_path, monkeypatch):
    from hybridix.indexing import index_versions

    (tmp_path / "version_002").mkdir()

    validator = Mock()
    monkeypatch.setattr(
        index_versions, "validate_index_pair", validator
    )

    chunks = [object()]

    index_versions.activate_index_version(
        "version_002",
        chunks,
        index_root=tmp_path,
    )

    assert index_versions.read_active_version(tmp_path) == "version_002"

    validator.assert_called_once_with(
        chunks,
        tmp_path / "version_002" / "dense",
        tmp_path / "version_002" / "sparse",
    )

def test_failed_validation_preserves_active_version(tmp_path, monkeypatch):
    from hybridix.indexing import index_versions

    (tmp_path / "version_002").mkdir()

    active_file = tmp_path / "active.json"
    active_file.write_text('{"version": "version_001"}')

    monkeypatch.setattr(
        index_versions,
        "validate_index_pair",
        Mock(side_effect=ValueError("Invalid indexes"))
)

    with pytest.raises(ValueError, match="Invalid indexes"):
        index_versions.activate_index_version(
            "version_002",
            [object()],
            index_root=tmp_path
        )

    assert index_versions.read_active_version(tmp_path) == "version_001"