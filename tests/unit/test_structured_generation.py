import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from pydantic import SecretStr
from hybridix.generation import generator
from hybridix.models import Chunk, ChunkMetadata, RetrievalResult

def make_result():
    return RetrievalResult(
        chunk=Chunk(
            content="X-RateLimit-Bucket identifies a rate-limit bucket.",
            metadata=ChunkMetadata(
                source="topics/rate-limits.mdx",
                file_type="mdx",
                chunk_id="test-chunk",
                strategy="heading",
                chunk_index=0,
                heading="Rate Limits"
            )
        ),
        score=0.9
    )

def mock_generation(monkeypatch, claims):
    client = MagicMock()

    client.chat.completions.create.return_value.choices[
        0
    ].message.content = json.dumps({"claims": claims})

    monkeypatch.setattr(
        generator,
        "Groq",
        MagicMock(return_value=client)
    )

    monkeypatch.setattr(
        generator,
        "settings",
        SimpleNamespace(
            groq_api_key=SecretStr("fake-key"),
            generation_model="test-model"
        )
    )

    return client

def test_structured_answer_resolves_citations(monkeypatch):
    client = mock_generation(
        monkeypatch,
        [
            {
                "text": "X-RateLimit-Bucket identifies the bucket.",
                "evidence_numbers": [1]
            }
        ]
    )

    answer = generator.generate_answer(
        "Which header identifies the bucket?",
        [make_result()]
    )

    assert answer.text == (
        "X-RateLimit-Bucket identifies the bucket. [1]"
    )
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "test-chunk"
    assert answer.citations[0].source == "topics/rate-limits.mdx"

    request = client.chat.completions.create.call_args.kwargs

    assert request["response_format"]["type"] == "json_schema"

def test_unknown_evidence_is_rejected(monkeypatch):
    mock_generation(
        monkeypatch,
        [
            {
                "text": "An unsupported claim.",
                "evidence_numbers": [99]
            }
        ]
    )

    with pytest.raises(ValueError, match="Unknown evidence number"):
        generator.generate_answer(
            "Test question",
            [make_result()]
        )

@pytest.mark.parametrize("marker", ["[1]", "[2]"])
def test_claim_text_cannot_inject_citations(monkeypatch, marker):
    mock_generation(
        monkeypatch,
        [
            {
                "text": f"The bucket header is X-RateLimit-Bucket {marker}.",
                "evidence_numbers": [1]
            }
        ]
    )

    with pytest.raises(ValueError, match="citation markers"):
        generator.generate_answer(
            "Which header identifies the bucket?",
            [make_result(), make_result()]
        )