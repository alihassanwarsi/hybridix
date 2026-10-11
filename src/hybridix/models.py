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
    chunk_id: str
    strategy: str
    chunk_index: int
    heading: str | None = None

@dataclass(slots=True)
class Chunk:
    content: str
    metadata: ChunkMetadata

@dataclass(slots=True)
class RetrievalResult:
    chunk: Chunk
    score: float

@dataclass(slots=True)
class Citation:
    number: int
    chunk_id: str
    source: str
    heading: str | None
    content: str

@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    citations: list[Citation]