from hybridix.ingestion.loader import load_mdx

def test_load_mdx_parses_title(tmp_path):
    path = tmp_path / "test.mdx"
    path.write_text(
        """---
title: Rate Limits
---

# Rate Limits

Hello world.
""",
        encoding="utf-8"
    )

    document = load_mdx(path, tmp_path)

    assert document.metadata.title == "Rate Limits"
    assert document.metadata.source == "test.mdx"
    assert "Hello world." in document.content

def test_loader_keeps_import_inside_code_block(tmp_path):
    path = tmp_path / "test.mdx"
    path.write_text(
        """import Component from "./Component"

# Example

```python
import requests
```
""",
        encoding="utf-8"
    )

    document = load_mdx(path, tmp_path)

    assert 'import Component from "./Component"' not in document.content
    assert "import requests" in document.content
