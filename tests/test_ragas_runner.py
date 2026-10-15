import json
from types import SimpleNamespace
from pydantic import SecretStr
from hybridix.evaluation import ragas_runner

def test_refusals_and_invalid_citations(tmp_path, monkeypatch):
    golden_cases = [
        {"id": "qa_032", "should_refuse": True},
        {"id": "qa_033", "should_refuse": True},
        {
            "id": "qa_002",
            "should_refuse": False,
            "reference_answer": "X-RateLimit-Bucket"
        }
    ]

    answers = [
        {
            "id": "qa_032",
            "question": "Unsupported question?",
            "generated_answer": "I don't have enough evidence to answer.",
            "citations": [],
            "error": None
        },
        {
            "id": "qa_033",
            "question": "Another unsupported question?",
            "generated_answer": "Definitely yes.",
            "citations": [],
            "error": None
        },
        {
            "id": "qa_002",
            "question": "Which header?",
            "generated_answer": "X-RateLimit-Bucket [99]",
            "citations": [{"number": 1, "content": "X-RateLimit-Bucket"}],
            "error": None
        }
    ]

    monkeypatch.setattr(
        ragas_runner,
        "load_eval_cases",
        lambda path: golden_cases,
    )

    monkeypatch.setattr(
        ragas_runner,
        "settings",
        SimpleNamespace(
            openai_api_key=SecretStr("fake-test-key"),
            evaluation_model="test-model"
        ),
    )

    for name in (
        "AsyncOpenAI",
        "llm_factory",
        "embedding_factory",
        "FactualCorrectness",
        "Faithfulness",
        "AnswerRelevancy"
    ):
        monkeypatch.setattr(
            ragas_runner,
            name,
            lambda *args, **kwargs: object()
        )

    monkeypatch.setattr(
        ragas_runner,
        "calculate_score",
        lambda *args, **kwargs: (1.0, None)
    )

    report_path = tmp_path / "answers.json"
    report_path.write_text(json.dumps({"results": answers}))

    ragas_runner.run_ragas_evaluation(report_path)

    saved_files = list(tmp_path.glob("ragas_results_*.json"))
    assert len(saved_files) == 1

    report = json.loads(saved_files[0].read_text())

    results = {
        row["id"]: row
        for row in report["results"]
    }

    assert results["qa_032"]["refusal_correct"] is True
    assert results["qa_033"]["refusal_correct"] is False
    assert results["qa_002"]["citations_valid"] is False
