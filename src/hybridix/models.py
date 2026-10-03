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