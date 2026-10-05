import yaml
from pathlib import Path
from typing import Any
from hybridix.models import Document, DocumentMetadata

def _parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    if not raw_text.startswith("---"):
        return {}, raw_text

    parts = raw_text.split("---", 2)

    if len(parts) < 3:
        raise ValueError("Invalid YAML frontmatter")

    raw_metadata = parts[1]
    body = parts[2].lstrip()

    metadata = yaml.safe_load(raw_metadata) or {}

    if not isinstance(metadata, dict):
        raise ValueError("Frontmatter must be a YAML mapping")

    return metadata, body

def load_mdx(path: Path, source_root: Path) -> Document:
    text = path.read_text(encoding="utf-8")

    metadata, body = _parse_frontmatter(text)

    source = path.resolve().relative_to(source_root.resolve()).as_posix()

    return Document(
        content=body,
        metadata=DocumentMetadata(
            source=source,
            file_type="mdx",
            title=metadata.get("title"),
        ),
    )