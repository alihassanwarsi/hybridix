from hybridix.models import Chunk, ChunkMetadata, Document

def _is_heading(line:str) -> bool:
    return line.startswith(("# ", "## ", "### ", "#### ", "##### ", "###### "))

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

        chunks.append(
            Chunk(
                content=content,
                metadata=ChunkMetadata(
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