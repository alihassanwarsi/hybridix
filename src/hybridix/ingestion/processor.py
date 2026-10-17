from pathlib import Path
from hybridix.models import Chunk
from hybridix.ingestion.loader import discover_documents, load_markdown, load_mdx
from hybridix.ingestion.chunking import chunk_by_headings

def prepare_chunks(source_root: Path) -> list[Chunk]:
    paths = discover_documents(source_root)

    if not paths:
        raise ValueError(f"No Markdown or MDX files found in {source_root}")

    chunks: list[Chunk] = []

    for path in paths:
        if path.suffix.lower() == ".md":
            document = load_markdown(path, source_root)
        else:
            document = load_mdx(path, source_root)

        chunks.extend(chunk_by_headings(document))

    if not chunks:
        raise ValueError("No chunks generated.")

    return chunks