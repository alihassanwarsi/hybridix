import pytest
from unittest.mock import Mock
from hybridix import pipeline
from hybridix.generation.evidence_verifier import VerificationResult
from hybridix.models import Citation, GeneratedAnswer

def make_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        text="Guild commands update instantly [1].",
        citations=[
            Citation(
                number=1,
                chunk_id="test-chunk",
                source="docs.mdx",
                heading="Guild Commands",
                content="Guild commands update instantly."
            )
        ]
    )

def test_pipeline_returns_verified_answer(monkeypatch):
    answer = make_answer()
    results = [object()]

    search = Mock(return_value=results)
    generate = Mock(return_value=answer)
    verify = Mock(
        return_value=VerificationResult(
            supported=True,
            reason="The evidence supports the answer."
        )
    )

    monkeypatch.setattr(pipeline, "reranked_search", search)
    monkeypatch.setattr(pipeline, "generate_answer", generate)
    monkeypatch.setattr(pipeline, "verify_answer", verify)

    output = pipeline.answer_question("How do guild commands update?")

    assert output.retrieved_evidence is results
    assert output is answer
    assert output.citations[0].chunk_id == "test-chunk"

    search.assert_called_once_with(
        "How do guild commands update?",
        top_k=5
    )
    generate.assert_called_once_with(
        "How do guild commands update?",
        results
    )
    verify.assert_called_once_with(answer)


def test_pipeline_rejects_unverified_answer(monkeypatch):
    answer = make_answer()

    monkeypatch.setattr(
        pipeline,
        "reranked_search",
        Mock(return_value=[object()])
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        Mock(return_value=answer)
    )
    monkeypatch.setattr(
        pipeline,
        "verify_answer",
        Mock(
            return_value=VerificationResult(
                supported=False,
                reason="The claim is unsupported."
            )
        )
    )

    output = pipeline.answer_question("Test question")

    assert output.text == "I couldn't verify an answer from the available evidence."
    assert output.citations == []
    assert output.refusal_reason == "verifier_rejection"
    assert output.verification_reason == "The claim is unsupported."
    assert output.unverified_answer == answer.text

def test_pipeline_handles_no_evidence(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "reranked_search",
        Mock(return_value=[])
    )

    generate = Mock()
    verify = Mock()

    monkeypatch.setattr(pipeline, "generate_answer", generate)
    monkeypatch.setattr(pipeline, "verify_answer", verify)

    output = pipeline.answer_question("Unknown question")

    assert output.text == "I don't have enough evidence to answer."
    assert output.citations == []
    assert output.refusal_reason == "no_results"

    generate.assert_not_called()
    verify.assert_not_called()


@pytest.mark.parametrize(
    "query, top_k",
    [
        ("", 5),
        ("   ", 5),
        ("Valid question", 0),
        ("Valid question", -1)
    ]
)
def test_pipeline_rejects_invalid_input(query, top_k):
    with pytest.raises(ValueError):
        pipeline.answer_question(query, top_k=top_k)

def test_pipeline_preserves_generator_refusal(monkeypatch):
    refusal = GeneratedAnswer(
        text="I don't have enough evidence to answer.",
        citations=[],
        refused=True
    )

    monkeypatch.setattr(
        pipeline,
        "reranked_search",
        Mock(return_value=[object()])
    )

    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        Mock(return_value=refusal),
    )

    verifier = Mock(
        return_value=VerificationResult(
            supported=False,
            reason="No cited evidence."
        )
    )

    monkeypatch.setattr(
        pipeline,
        "verify_answer",
        verifier
    )

    result = pipeline.answer_question("Unknown question")

    assert result is refusal
    assert result.refused is True
    assert result.refusal_reason == "generator_refusal"
    verifier.assert_not_called()


def test_build_indexes_builds_before_activation(tmp_path, monkeypatch):
    source_root = tmp_path / "docs"
    source_root.mkdir()
    (source_root / "guide.mdx").write_text("# Hello")

    index_root = tmp_path / "indexes"
    chunks = [object()]
    steps = []

    monkeypatch.setattr(
        pipeline,
        "prepare_chunks",
        lambda root: chunks
    )
    monkeypatch.setattr(
        pipeline, "fit_chunks_to_embedding_budget", lambda items: items
    )

    def fake_build(items, version, *, index_root):
        steps.append(("build", version, index_root))

    def fake_activate(version, items, *, index_root):
        steps.append(("activate", version, index_root))

    monkeypatch.setattr(pipeline, "build_index_version", fake_build)
    monkeypatch.setattr(pipeline, "activate_index_version", fake_activate)

    count = pipeline.build_indexes(
        source_root,
        version="version_test",
        index_root=index_root,
    )

    assert count == 1
    assert steps == [
        ("build", "version_test", index_root),
        ("activate", "version_test", index_root),
    ]


def test_failed_build_never_activates(tmp_path, monkeypatch):
    source_root = tmp_path / "docs"
    source_root.mkdir()
    (source_root / "guide.mdx").write_text("# Hello")

    monkeypatch.setattr(
        pipeline,
        "prepare_chunks",
        lambda root: [object()]
    )
    monkeypatch.setattr(
        pipeline,
        "fit_chunks_to_embedding_budget",
        lambda chunks: chunks
    )

    monkeypatch.setattr(
        pipeline,
        "build_index_version",
        Mock(side_effect=RuntimeError("Index build failed"))
    )

    activate = Mock()
    monkeypatch.setattr(
        pipeline, "activate_index_version", activate
    )

    with pytest.raises(RuntimeError, match="Index build failed"):
        pipeline.build_indexes(
            source_root,
            version="version_test",
            index_root=tmp_path / "indexes"
        )

    activate.assert_not_called()