from hashlib import sha256
from dataclasses import replace
from hybridix.models import Chunk, RetrievalResult
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

def merge_adjacent_results(results: list[RetrievalResult]) -> list[RetrievalResult]:

    remaining = list(results)
    merged = []

    while remaining:
        group = [remaining.pop(0)]

        changed = True
        while changed:
            changed = False

            for candidate in remaining[:]:
                meta = candidate.chunk.metadata

                if any(
                    meta.source == item.chunk.metadata.source
                    and meta.heading == item.chunk.metadata.heading
                    and meta.strategy == item.chunk.metadata.strategy
                    and abs(meta.chunk_index - item.chunk.metadata.chunk_index) == 1
                    for item in group
                ):
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True

        if len(group) == 1:
            merged.append(group[0])
            continue

        group.sort(key=lambda item: item.chunk.metadata.chunk_index)

        content = group[0].chunk.content

        for item in group[1:]:
            next_content = item.chunk.content

            overlap = next(
                (
                    size
                    for size in range(min(len(content), len(next_content)), 31, -1)
                    if content.endswith(next_content[:size])
                ),
                0
            )

            content += next_content[overlap:]

        ids = ":".join(item.chunk.metadata.chunk_id for item in group)
        merged_id = sha256(ids.encode("utf-8")).hexdigest()[:16]

        merged.append(
            RetrievalResult(
                chunk=Chunk(
                    content=content,
                    metadata=replace(
                        group[0].chunk.metadata,
                        chunk_id=merged_id,
                    ),
                ),
                score=max(item.score for item in group),
            )
        )

    return merged
