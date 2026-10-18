from hybridix.models import Chunk, ChunkMetadata, RetrievalResult
from hybridix.generation.context import build_context, merge_adjacent_results

def test_build_context_formats_retrieved_chunks():
    chunk = Chunk(
        content="Global commands are available on all your app's guilds.",
        metadata=ChunkMetadata(
            source="interactions/application-commands.mdx",
            file_type="mdx",
            title="Application Commands",
            heading="Making a Global Command",
            chunk_id="test-id",
            strategy="heading",
            chunk_index=0
        )
    )

    results = [
        RetrievalResult(chunk=chunk, score=1.0)
    ]

    context = build_context(results)

    assert context == (
    "[1] Source: interactions/application-commands.mdx "
    "| Section: Making a Global Command\n"
    "Global commands are available on all your app's guilds."
)

def test_merge_adjacent_chunks():
    def make_result(index, content):
        return RetrievalResult(
            chunk=Chunk(
                content=content,
                metadata=ChunkMetadata(
                    source="gateway.mdx",
                    file_type="mdx",
                    heading="Identifying",
                    chunk_id=f"chunk-{index}",
                    strategy="heading",
                    chunk_index=index
                )
            ),
            score=1.0,
        )

    results = [
        make_result(25, "Clients are limited to 1000 "),
        make_result(26, "IDENTIFY calls per 24 hours.")
    ]

    merged = merge_adjacent_results(results)

    assert len(merged) == 1
    assert "1000" in merged[0].chunk.content
    assert "24 hours" in merged[0].chunk.content

def test_different_headings_do_not_merge():
    results = []

    for index, heading in enumerate(["Identifying", "Resuming"]):
        chunk = Chunk(
            content=f"Content for {heading}",
            metadata=ChunkMetadata(
                source="gateway.mdx",
                file_type="mdx",
                heading=heading,
                chunk_id=f"chunk-{index}",
                strategy="heading",
                chunk_index=index
            )
        )

        results.append(RetrievalResult(chunk=chunk, score=1.0))

    merged = merge_adjacent_results(results)

    assert len(merged) == 2
