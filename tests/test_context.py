from hybridix.generation.context import build_context
from hybridix.models import Chunk, ChunkMetadata, RetrievalResult

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

    assert "Source: interactions/application-commands.mdx" in context
    assert "Section: Making a Global Command" in context
    assert "Global commands are available" in context