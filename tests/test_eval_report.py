
import json
from hybridix.evaluation import runner
from hybridix.models import Chunk, ChunkMetadata, RetrievalResult

def test_retrieval_report_is_saved(tmp_path, monkeypatch):
    result = RetrievalResult(
        chunk=Chunk(
            content="Example documentation.",
            metadata=ChunkMetadata(
                source="correct.mdx",
                file_type="mdx",
                chunk_id="test-chunk",
                strategy="heading",
                chunk_index=0,
                heading="Example"
            )
        ),
        score=0.9,
    )

    def fake_search(query, top_k):
        return [result]

    for name in (
        "dense_search",
        "sparse_search",
        "hybrid_search",
        "reranked_search"
    ):
        monkeypatch.setattr(runner, name, fake_search)

    cases = [
        {
            "query": "Matching question",
            "relevant": [
                {"source": "correct.mdx", "heading": "Example"}
            ]
        },
        {
            "query": "Missing question",
            "relevant": [
                {"source": "missing.mdx", "heading": "Other"}
            ]
        }
    ]

    dataset = tmp_path / "retrieval.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(case) for case in cases),
        encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    runner.run_retrieval_evaluation(dataset, k=5)

    reports = list((tmp_path / "data/eval/reports").glob("retrieval_*.json"))

    assert len(reports) == 1

    report = json.loads(reports[0].read_text(encoding="utf-8"))

    assert report["total_questions"] == 2
    assert len(report["retrievers"]) == 4

    for evaluation in report["retrievers"].values():
        assert evaluation["metrics"]["hit@5"] == 0.5
        assert evaluation["metrics"]["mrr@5"] == 0.5

        questions = evaluation["questions"]
        assert questions[0]["hit"] is True
        assert questions[1]["hit"] is False
        assert questions[0]["retrieved"][0]["chunk_id"] == "test-chunk"
