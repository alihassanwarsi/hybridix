import pytest
from unittest.mock import Mock
from pathlib import Path
from hybridix import cli, pipeline
from hybridix.evaluation import runner
from hybridix.models import Citation, GeneratedAnswer, Chunk, ChunkMetadata, RetrievalResult

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
            verification_reason="The claim is unsupported.",
            unverified_answer="Discord allows 5,000 IDENTIFY calls per day. [1]"
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
    assert "Unverified answer:" in output
    assert "Discord allows 5,000 IDENTIFY calls per day. [1]" in output

def test_ask_hides_unverified_answer_without_debug(monkeypatch, capsys):
    def fake_answer_question(query, top_k):
        return GeneratedAnswer(
            text="I couldn't verify an answer.",
            citations=[],
            refused=True,
            refusal_reason="verifier_rejection",
            verification_reason="The claim is unsupported.",
            unverified_answer="Incorrect answer [1]"
        )

    monkeypatch.setattr(
        pipeline,
        "answer_question",
        fake_answer_question
    )

    cli.main(["ask", "Unknown question"])

    output = capsys.readouterr().out

    assert "I couldn't verify an answer." in output
    assert "Incorrect answer [1]" not in output
    assert "Debug:" not in output

def test_debug_shows_retrieved_evidence(monkeypatch, capsys):
    evidence = RetrievalResult(
        chunk=Chunk(
            content="X-RateLimit-Bucket identifies the bucket.",
            metadata=ChunkMetadata(
                source="topics/rate-limits.mdx",
                file_type="mdx",
                chunk_id="test-chunk",
                strategy="heading",
                chunk_index=0,
                heading="Rate Limit Header Examples"
            )
        ),
        score=0.95
    )

    def fake_answer_question(query, top_k):
        return GeneratedAnswer(
            text="I don't have enough evidence to answer.",
            citations=[],
            refused=True,
            refusal_reason="generator_refusal",
            retrieved_evidence=[evidence]
        )

    monkeypatch.setattr(
        pipeline,
        "answer_question",
        fake_answer_question
    )

    cli.main(["ask", "Test question", "--debug"])

    output = capsys.readouterr().out

    assert "Retrieved evidence:" in output
    assert "topics/rate-limits.mdx" in output
    assert "Rate Limit Header Examples" in output

def test_ingest_command(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "guide.mdx").write_text(
        "# Test documentation",
        encoding="utf-8"
    )

    build = Mock(return_value=3)
    monkeypatch.setattr(pipeline, "build_indexes", build)

    result = cli.main(["ingest", str(docs)])

    assert result == 0
    build.assert_called_once_with(docs)

    output = capsys.readouterr().out
    assert "Successfully indexed" in output
    assert "3 chunks" in output

def test_ingest_dry_run_does_not_build(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "guide.mdx").write_text(
        "# Test documentation",
        encoding="utf-8"
    )

    build = Mock()
    monkeypatch.setattr(pipeline, "build_indexes", build)

    result = cli.main([
        "ingest",
        str(docs),
        "--dry-run"
    ])

    assert result == 0
    build.assert_not_called()

    output = capsys.readouterr().out
    assert "Found 1 supported document(s)." in output
    assert "No indexes were modified" in output

def test_ingest_rejects_missing_folder(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "ingest",
            str(tmp_path / "missing")
        ])

    assert exc.value.code == 2

def test_ingest_dry_run_detects_both_formats(tmp_path, monkeypatch, capsys):
    from unittest.mock import Mock

    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "guide.md").write_text("# Markdown")
    (docs / "guide.mdx").write_text("# MDX")

    build = Mock()
    monkeypatch.setattr(pipeline, "build_indexes", build)

    result = cli.main(["ingest", str(docs), "--dry-run"])

    assert result == 0
    build.assert_not_called()

    output = capsys.readouterr().out
    assert "Found 2 supported document(s)." in output