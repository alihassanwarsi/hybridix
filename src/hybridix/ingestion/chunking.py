from hashlib import sha256
from hybridix.models import Chunk, ChunkMetadata, Document

def _is_heading(line:str) -> bool:
    return line.startswith(("# ", "## ", "### ", "#### ", "##### ", "###### "))

def _generate_chunk_id(source: str, strategy: str, heading: str | None, content: str, chunk_index: int) -> str:
    raw = f"{source}:{strategy}:{heading}:{chunk_index}:{content}"
    return sha256(raw.encode("utf-8")).hexdigest()[:16]

def _split_large_section(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sections = []
    start = 0

    while start < len(text):
        end = start + max_chars
        section = text[start:end].strip()

        if section:
            sections.append(section)

        start = end - overlap

    return sections

def chunk_by_headings(document: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_heading: str | None = None
    inside_code_block = False

    def save_chunk() -> None:
        if not current_lines:
            return

        content = "\n".join(current_lines).strip()

        if not content:
            return

        sections = _split_large_section(content)
        for section in sections:
            chunks.append(
                Chunk(
                    content=section,
                    metadata=ChunkMetadata(
                        chunk_id=_generate_chunk_id(
                            source=document.metadata.source,
                            strategy="heading",
                            heading=current_heading,
                            content=section,
                            chunk_index = len(chunks)
                        ),
                        source=document.metadata.source,
                        file_type=document.metadata.file_type,
                        title=document.metadata.title,
                        strategy="heading",
                        chunk_index=len(chunks),
                        heading=current_heading
                    )
                )
            )

    for line in document.content.splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            inside_code_block = not inside_code_block
            current_lines.append(line)
            continue

        if not inside_code_block and _is_heading(stripped):
            save_chunk()
            current_lines.clear()

            current_heading = stripped.lstrip("#").strip()

        current_lines.append(line)

    save_chunk()

    return chunks