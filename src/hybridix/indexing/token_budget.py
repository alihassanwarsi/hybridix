from dataclasses import replace
from hashlib import sha256
from hybridix.indexing.embeddings import get_embedding_model
from hybridix.indexing.search_text import build_search_text
from hybridix.models import Chunk

def fit_chunks_to_embedding_budget(chunks: list[Chunk]) -> list[Chunk]:
    if not chunks:
        return []

    model = get_embedding_model()
    tokenizer = model.tokenizer
    max_tokens = int(model.max_seq_length)

    if max_tokens <= 0:
        raise ValueError("Embedding model has an invalid token limit.")

    def token_count(chunk: Chunk) -> int:
        text = build_search_text(chunk)
        return len(tokenizer.encode(text, add_special_tokens=True))

    fitted: list[Chunk] = []

    for chunk in chunks:
        if token_count(chunk) <= max_tokens:
            fitted.append(chunk)
            continue

        text = chunk.content
        start = 0

        if not text:
            raise ValueError(f"Embedding metadata exceeds token limit: {chunk.metadata.source}")

        while start < len(text):
            left = start + 1
            right = len(text)
            best_end = start

            while left <= right:
                middle = (left + right) // 2

                candidate = Chunk(
                    content=text[start:middle],
                    metadata=chunk.metadata
                )

                if token_count(candidate) <= max_tokens:
                    best_end = middle
                    left = middle  + 1
                else:
                    right = middle - 1

            if best_end == start:
                raise ValueError(f"Not enough embedding tokens for content in {chunk.metadata.source}. Check the document title and heading.")

            end = best_end

            if end < len(text):
                midpoint = start + (end - start) // 2

                break_at = max(
                    text.rfind("\n", midpoint, end),
                    text.rfind(" ", midpoint, end),
                    text.rfind("\t", midpoint, end)
                )

                if break_at >= midpoint:
                    end = break_at + 1

            piece = text[start:end]

            new_id = sha256(f"{chunk.metadata.chunk_id}:{start}:{piece}".encode("utf-8")).hexdigest()[:16]

            fitted.append(
                Chunk(
                    content=piece,
                    metadata=replace(
                        chunk.metadata,
                        chunk_id=new_id
                    )
                )
            )

            start = end

    next_index_by_source: dict[str, int] = {}
    result: list[Chunk] = []

    for chunk in fitted:
            source = chunk.metadata.source
            index = next_index_by_source.get(source, 0)
            next_index_by_source[source] = index + 1

            result.append(
                Chunk(
                    content=chunk.content,
                    metadata=replace(
                        chunk.metadata,
                        chunk_index=index
                    )
                )
            )

    return result