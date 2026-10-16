from pathlib import Path
import pytest
from hybridix import cli, pipeline
from hybridix.evaluation import runner
from hybridix.models import Citation, GeneratedAnswer

def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0

    output = capsys.readouterr().out
    assert "evaluate" in output

def test_evaluate_command(monkeypatch):
    calls = []

    def fake_evaluation(path, k):
        calls.append((path, k))

    monkeypatch.setattr(
        runner,
        "run_retrieval_evaluation",
        fake_evaluation
    )

    exit_code = cli.main([
        "evaluate",
        "--dataset", "custom.jsonl",
        "--top-k", "3"
    ])

    assert exit_code == 0
    assert calls == [(Path("custom.jsonl"), 3)]

def test_evaluate_rejects_invalid_top_k():
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "--top-k", "0"])

    assert exc.value.code == 2

def test_ask_command(monkeypatch, capsys):
    calls = []

    def fake_answer_question(query, top_k):
        calls.append((query, top_k))

        return GeneratedAnswer(
            text="Discord uses rate-limit headers. [1]",
            citations=[
                Citation(
                    number=1,
                    chunk_id="test-chunk",
                    source="rate-limits.mdx",
                    heading="Rate Limits",
                    content="Example documentation."
                )
            ]
        )

    monkeypatch.setattr(
        pipeline,
        "answer_question",
        fake_answer_question,
    )

    exit_code = cli.main([
        "ask",
        "How do rate limits work?",
        "--top-k", "3"
    ])

    output = capsys.readouterr().out

    assert exit_code == 0
    assert calls == [("How do rate limits work?", 3)]

    assert "Discord uses rate-limit headers." in output
    assert "rate-limits.mdx" in output
    assert "Rate Limits" in output


def test_ask_rejects_invalid_top_k():
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "ask",
            "Test question",
            "--top-k", "0"
        ])

    assert exc.value.code == 2

def test_ask_debug_shows_refusal_details(monkeypatch, capsys):
    def fake_answer_question(query, top_k):
        return GeneratedAnswer(
            text="I couldn't verify an answer.",
            citations=[],
            refused=True,
            refusal_reason="verifier_rejection",
            verification_reason="The claim is unsupported."
        )

    monkeypatch.setattr(
        pipeline,
        "answer_question",
        fake_answer_question,
    )

    cli.main(["ask", "Unknown question", "--debug"])

    output = capsys.readouterr().out

    assert "verifier_rejection" in output
    assert "The claim is unsupported." in output