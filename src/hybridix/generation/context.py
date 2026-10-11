from hybridix.models import RetrievalResult

def build_context(results: list[RetrievalResult]) -> str:
    sections = []

    for number, result in enumerate(results, start=1):
        metadata = result.chunk.metadata
        
        header = f"[{number}] Source: {metadata.source}"

        if metadata.heading:
            header += f" | Section: {metadata.heading}"

        sections.append(f"{header}\n{result.chunk.content}")

    return"\n\n".join(sections)