import pytest
from unittest.mock import Mock
from hybridix.indexing.versioning import builder as version_builder

def test_build_version_runs_in_correct_order(tmp_path, monkeypatch):
    steps = []

    def fake_dense(chunks, *, index_path):
        steps.append(("dense", index_path))

    def fake_sparse(chunks, *, index_path):
        steps.append(("sparse", index_path))

    def fake_validate(chunks, dense_path, sparse_path):
        steps.append(("validate", dense_path, sparse_path))

    monkeypatch.setattr(version_builder, "build_dense_index", fake_dense)
    monkeypatch.setattr(version_builder, "build_sparse_index", fake_sparse)
    monkeypatch.setattr(version_builder, "validate_index_pair", fake_validate)

    result = version_builder.build_index_version(
        chunks=[object()],
        version="version_002",
        index_root=tmp_path
    )

    assert result == tmp_path / "version_002"
    assert steps == [
        ("dense", result / "dense"),
        ("sparse", result / "sparse"),
        ("validate", result / "dense", result / "sparse")
    ]

    with pytest.raises(FileExistsError):
        version_builder.build_index_version(
            chunks=[object()],
            version="version_002",
            index_root=tmp_path,
        )

def test_failed_build_does_not_change_active_version(tmp_path, monkeypatch):
    active_file = tmp_path / "active.json"
    active_file.write_text('{"version": "version_001"}')

    monkeypatch.setattr(
        version_builder,
        "build_dense_index",
        Mock()
    )

    monkeypatch.setattr(
        version_builder,
        "build_sparse_index",
        Mock(side_effect=RuntimeError("BM25 build failed"))
    )

    validate = Mock()
    monkeypatch.setattr(version_builder, "validate_index_pair", validate)

    with pytest.raises(RuntimeError, match="BM25 build failed"):
        version_builder.build_index_version(
            chunks=[object()],
            version="version_002",
            index_root=tmp_path
        )

    assert active_file.read_text() == '{"version": "version_001"}'
    validate.assert_not_called()