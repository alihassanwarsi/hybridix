from hybridix.models import Chunk

def build_search_text(chunk: Chunk) -> str:
    parts = []

    if chunk.metadata.title:
        parts.append(f"Title: {chunk.metadata.title}")

    if chunk.metadata.heading:
        parts.append(f"Section: {chunk.metadata.heading}")

    parts.append(chunk.content)

    return "\n\n".join(parts)