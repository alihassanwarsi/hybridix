from hybridix.config import settings
from hybridix.ingestion.chunking import chunk_by_headings
from hybridix.models import Document, DocumentMetadata

def make_document(content: str) -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(
            source="test.mdx",
            file_type="mdx",
            title="Test",
        )
    )

def test_chunking_splits_on_headings():
    document = make_document(
        """# First

First section.

## Second

Second section.
"""
    )

    chunks = chunk_by_headings(document)

    assert len(chunks) == 2
    assert chunks[0].metadata.heading == "First"
    assert chunks[1].metadata.heading == "Second"

def test_oversized_chunks_are_split():
    document = make_document(
        "# Large\n\n" + ("a" * (settings.chunk_size + 500))
    )

    chunks = chunk_by_headings(document)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= settings.chunk_size for chunk in chunks)

def test_chunk_ids_are_deterministic():
    document = make_document(
        """# Example

Some content.
"""
    )

    first = chunk_by_headings(document)
    second = chunk_by_headings(document)

    assert first[0].metadata.chunk_id == second[0].metadata.chunk_id
