from hybridix.indexing.search_text import build_search_text
from hybridix.models import Chunk, ChunkMetadata

def make_chunk(*, content: str = "some content", title: str | None = "Gateway", heading: str | None = "Sharding") -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            source="events/gateway.mdx",
            file_type="mdx",
            title=title,
            heading=heading,
            chunk_id="test-id",
            strategy="heading",
            chunk_index=0
        )
    )

def test_build_index_text_adds_context():
    chunk = make_chunk(content="shard_id = guild_id >> 22")

    text = build_search_text(chunk)

    assert "Gateway" in text
    assert "Sharding" in text
    assert "shard_id = guild_id >> 22" in text

def test_build_index_text_handles_missing_metadata():
    chunk = make_chunk(
        content="hello",
        title=None,
        heading=None,
    )

    text = build_search_text(chunk)

    assert text == "hello"