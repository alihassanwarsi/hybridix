from dataclasses import dataclass

@dataclass(slots=True)
class DocumentMetadata:
    source: str
    file_type: str
    title: str | None = None

@dataclass(slots=True)
class Document:
    content: str
    metadata: DocumentMetadata

@dataclass(slots=True, kw_only=True)
class ChunkMetadata(DocumentMetadata):
    strategy: str
    chunk_index: int
    heading: str | None = None

@dataclass(slots=True)
class Chunk:
    content: str
    metadata: ChunkMetadata
