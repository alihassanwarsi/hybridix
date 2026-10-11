from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from pydantic import SecretStr
from hybridix.generation import evidence_verifier
from hybridix.models import Citation, GeneratedAnswer

def make_answer(text: str) -> GeneratedAnswer:
    return GeneratedAnswer(
        text=text,
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

@pytest.mark.parametrize(
    "llm_response, expected",
    [
        ('{"supported": true, "reason": "The evidence supports the claim."}', True),
        ('{"supported": false, "reason": "The evidence contradicts the claim."}', False)
    ]
)
def test_verify_answer(monkeypatch, llm_response, expected):
    client = MagicMock()

    client.chat.completions.create.return_value.choices[0].message.content = llm_response

    monkeypatch.setattr(
        evidence_verifier,
        "Groq",
        MagicMock(return_value=client)
    )

    monkeypatch.setattr(
        evidence_verifier,
        "settings",
        SimpleNamespace(
            groq_api_key=SecretStr("fake-key"),
            generation_model="test-model"
        )
    )

    answer = make_answer("Guild commands update instantly [1].")

    result = evidence_verifier.verify_answer(answer)

    assert result.supported is expected
    assert result.reason

    messages = client.chat.completions.create.call_args.kwargs["messages"]

    assert answer.text in messages[1]["content"]
    assert "Guild commands update instantly." in messages[1]["content"]

def test_verify_answer_without_citations():
    answer = GeneratedAnswer(text="Unsupported answer.", citations=[])

    result = evidence_verifier.verify_answer(answer)

    assert result.supported is False
    assert result.reason == "No cited evidence to verify."