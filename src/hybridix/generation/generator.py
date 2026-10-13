from groq import Groq
from pydantic import BaseModel, ConfigDict, StrictInt
from hybridix.config import settings
from hybridix.generation.citations import resolve_citations
from hybridix.generation.context import build_context
from hybridix.generation.prompts import load_prompt
from hybridix.models import GeneratedAnswer, RetrievalResult

SYSTEM_PROMPT = load_prompt("answer_generation.txt")

class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_numbers: list[StrictInt]

class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[AnswerClaim]

def generate_answer(query: str, results: list[RetrievalResult]) -> GeneratedAnswer:
    if settings.groq_api_key is None:
        raise ValueError("Groq API key is not configured.")

    if not results:
        raise ValueError("No retrieved evidence provided.")

    context = build_context(results)

    client = Groq(api_key=settings.groq_api_key.get_secret_value())

    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}"
                )
            }
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_answer",
                "strict": True,
                "schema": StructuredAnswer.model_json_schema()
            }
        }
    )

    answer = response.choices[0].message.content or ""

    if not answer:
        raise ValueError("Empty generation response.")

    structured = StructuredAnswer.model_validate_json(answer)

    if not structured.claims:
        return GeneratedAnswer(
            text="I don't have enough evidence to answer.",
            citations=[]
        )

    sections = []

    for claim in structured.claims:
        text = claim.text.strip()
        numbers = list(dict.fromkeys(claim.evidence_numbers))

        if not text:
            raise ValueError("Generated claim cannot be empty.")

        if not numbers:
            raise ValueError("Generated claim has no evidence.")

        for number in numbers:
            if number < 1 or number > len(results):
                raise ValueError(f"Unknown evidence number: {number}")

        citation_text = " ".join(f"[{number}]" for number in numbers)

        sections.append(f"{text} {citation_text}")

    answer = "\n\n".join(sections)

    citations = resolve_citations(answer, results, require_citations=True)

    return GeneratedAnswer(text=answer, citations=citations)
