from groq import Groq
from pydantic import BaseModel, ConfigDict
from hybridix.config import settings
from hybridix.generation.prompts import load_prompt
from hybridix.models import GeneratedAnswer

SYSTEM_PROMPT = load_prompt("evidence_verification.txt")

class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported: bool
    reason: str

def verify_answer(answer: GeneratedAnswer) -> VerificationResult:
    if not answer.citations:
        return VerificationResult(supported=False, reason="No cited evidence to verify.")

    if settings.groq_api_key is None:
            raise ValueError("Groq API key is not configured.")
    
    evidence = "\n\n".join(
        f"[{citation.number}] Source: {citation.source}\n"
        f"{citation.content}"
        for citation in answer.citations
    )

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
                        f"Answer:\n{answer.text}\n\n"
                        f"Cited evidence:\n{evidence}"
                   )
              }
         ],
         temperature=0,
         response_format={
              "type": "json_schema",
              "json_schema": {
                   "name": "verification_result",
                   "strict": True,
                   "schema": VerificationResult.model_json_schema()
              }
         }
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Empty verification response.")

    return VerificationResult.model_validate_json(content)