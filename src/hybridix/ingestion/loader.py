import re
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

def _normalize_mdx(text: str) -> str:

    # Remove MDX imports, but preserve imports inside code blocks
    lines = []
    inside_code_block = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            inside_code_block = not inside_code_block
            lines.append(line)
            continue

        if not inside_code_block and stripped.startswith("import "):
            continue

        lines.append(line)

    text = "\n".join(lines)

    # <Route method="GET">/gateway</Route>
    # -> GET /gateway
    text = re.sub(
        r'<Route\s+method="([^"]+)">\s*(.*?)\s*</Route>',
        r"\1 \2",
        text,
    )

    # Remove wrapper tags but keep their content
    for tag in ("Info", "Warning", "Note", "Danger", "Tip"):
        text = text.replace(f"<{tag}>", "")
        text = text.replace(f"</{tag}>", "")

    # Remove navigation-only anchors
    text = re.sub(r"<ManualAnchor[^>]*/>", "", text)

    return text.strip()

def load_mdx(path: Path, source_root: Path) -> Document:
    text = path.read_text(encoding="utf-8")

    metadata, body = _parse_frontmatter(text)
    body = _normalize_mdx(body)

    source = path.resolve().relative_to(source_root.resolve()).as_posix()

    return Document(
        content=body,
        metadata=DocumentMetadata(
            source=source,
            file_type="mdx",
            title=metadata.get("title")
        )
    )