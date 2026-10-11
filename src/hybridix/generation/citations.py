import re
from hybridix.models import Citation, RetrievalResult

def resolve_citations(answer: str, results: list[RetrievalResult], require_citations: bool = False) -> list[Citation]:
    numbers = re.findall(r"\[(\d+)\]", answer)

    citations = []
    seen = set()

    for raw_number in numbers:
        number = int(raw_number)

        if number < 1 or number > len(results):
            raise ValueError(f"Unknown citation [{number}]")

        if number in seen:
            continue

        seen.add(number)

        chunk = results[number - 1].chunk
        metadata = chunk.metadata

        citations.append(
            Citation(
                number=number,
                chunk_id=metadata.chunk_id,
                source=metadata.source,
                heading=metadata.heading,
                content=chunk.content
            )
        )

    if require_citations and not citations:
        raise ValueError("Answer is missing citations.")

    return citations