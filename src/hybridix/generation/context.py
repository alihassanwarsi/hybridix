from hybridix.models import RetrievalResult

def build_context(results: list[RetrievalResult]) -> str:
    sections = []

    for result in results:
        metadata = result.chunk.metadata
        
        header = f"Source: {metadata.source}"

        if metadata.heading:
            header += f" | Section: {metadata.heading}"

        sections.append(f"[{header}]\n{result.chunk.content}")

    return"\n\n".join(sections)