from groq import Groq
from hybridix.config import settings
from hybridix.generation.context import build_context
from hybridix.generation.prompts import SYSTEM_PROMPT
from hybridix.models import RetrievalResult

def generate_answer(query: str, results: list[RetrievalResult]) -> str:
    if settings.groq_api_key is None:
        raise ValueError("Groq API key is not configured.")

    context = build_context(results)
    client = Groq(api_key=settings.groq_api_key.get_secret_value())

    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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
    )

    return response.choices[0].message.content or ""