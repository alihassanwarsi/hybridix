import pytest
from pydantic import ValidationError
from hybridix.config import Settings
from hybridix.config import settings
from hybridix.ingestion.chunking import chunk_by_headings
from hybridix.models import Document, DocumentMetadata
from hybridix.ingestion.chunking import _split_large_section

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

@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101)
    ]
)

def test_invalid_chunk_settings_are_rejected(chunk_size, chunk_overlap):
    with pytest.raises(ValidationError):
        Settings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

def test_split_large_section_avoids_redundant_tail():
    sections = _split_large_section(
        "abcdefghij",
        max_chars=6,
        overlap=2
    )

    assert sections == ["abcdef", "efghij"]

@pytest.mark.parametrize(
    "max_chars, overlap",
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6)
    ]
)
def test_split_large_section_rejects_invalid_arguments(max_chars, overlap):
    with pytest.raises(ValueError):
        _split_large_section(
            "",
            max_chars=max_chars,
            overlap=overlap
        )