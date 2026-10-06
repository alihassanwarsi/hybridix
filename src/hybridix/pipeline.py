from pathlib import Path
from hybridix.ingestion.chunking import chunk_by_headings
from hybridix.ingestion.loader import load_mdx
from hybridix.indexing.dense_indexer import build_dense_index
from hybridix.indexing.sparse_indexer import build_sparse_index

def build_indexes(source_root: Path) -> int:
    paths = sorted(source_root.rglob("*.mdx"))

    if not paths:
        raise ValueError(f"No MDX files found in {source_root}")

    chunks = []

    for path in paths:
        document = load_mdx(path, source_root)
        chunks.extend(chunk_by_headings(document))

    if not chunks:
        raise ValueError("No chunks generated.")

    build_dense_index(chunks)
    build_sparse_index(chunks)

    return len(chunks)