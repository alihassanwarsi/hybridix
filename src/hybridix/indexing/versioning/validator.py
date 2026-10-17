import bm25s
from bm25s.tokenization import Tokenizer
from pathlib import Path
from hybridix.config import settings
from hybridix.models import Chunk
from hybridix.indexing.dense_indexer import get_client


def validate_index_pair(chunks: list[Chunk], dense_path: Path, sparse_path: Path,) -> None:
    expected_ids = [chunk.metadata.chunk_id for chunk in chunks]

    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Expected chunks must have unique, non-empty IDs.")

    if not (dense_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Dense index missing: {dense_path}")

    if not (sparse_path / "params.index.json").is_file():
        raise FileNotFoundError(f"BM25 index missing: {sparse_path}")

    client = get_client(dense_path)
    collection = client.get_collection(name=settings.chroma_collection_name)
    dense_ids = collection.get(include=[])["ids"]

    sparse_index = bm25s.BM25.load(str(sparse_path), load_corpus=True)
    sparse_ids = [document["id"] for document in sparse_index.corpus]

    tokenizer = Tokenizer()
    tokenizer.load_vocab(str(sparse_path))
    tokenizer.load_stopwords(str(sparse_path))

    for name, ids in [
        ("Dense", dense_ids),
        ("BM25", sparse_ids)
    ]:
        if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
            raise ValueError(f"{name} index chunk IDs do not match expected chunks.")