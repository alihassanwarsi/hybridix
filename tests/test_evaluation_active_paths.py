import json
from pathlib import Path
from unittest.mock import Mock
from hybridix.evaluation import runner

def test_evaluation_uses_active_index_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dense_path = tmp_path / "version_002" / "dense"
    sparse_path = tmp_path / "version_002" / "sparse"

    resolver = Mock(return_value=(dense_path, sparse_path))
    dense = Mock(return_value=[])
    sparse = Mock(return_value=[])

    monkeypatch.setattr(
        runner, "get_active_index_paths", resolver
    )
    monkeypatch.setattr(
        runner,
        "load_eval_cases",
        lambda path: [{"query": "rate limits", "relevant": []}]
    )
    monkeypatch.setattr(runner, "dense_search", dense)
    monkeypatch.setattr(runner, "sparse_search", sparse)
    monkeypatch.setattr(
        runner, "hybrid_search", Mock(return_value=[])
    )
    monkeypatch.setattr(
        runner, "reranked_search", Mock(return_value=[])
    )

    runner.run_retrieval_evaluation(Path("test.json"))

    resolver.assert_called_once_with()

    dense.assert_called_once_with(
        "rate limits", top_k=5, index_path=dense_path
    )
    sparse.assert_called_once_with(
        "rate limits", top_k=5, index_path=sparse_path
    )

    report_dir = tmp_path / "data" / "eval" / "reports"
    report_file = next(report_dir.glob("retrieval_*.json"))

    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert report["config"]["dense_index_path"] == str(dense_path)
    assert report["config"]["sparse_index_path"] == str(sparse_path)