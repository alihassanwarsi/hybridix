import bm25s
from bm25s.tokenization import Tokenizer
from pathlib import Path
from hybridix.config import settings
from hybridix.models import Chunk
from hybridix.indexing.search_text import build_search_text

def build_sparse_index(chunks: list[Chunk], *, index_path: Path | None = None) -> None:
    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    if index_path is not None and index_path.exists():
        raise FileExistsError(f"Index destination already exists: {index_path}")

    corpus = [
    {
        "id": chunk.metadata.chunk_id,
        "text": chunk.content,
        "source": chunk.metadata.source,
        "file_type": chunk.metadata.file_type,
        "title": chunk.metadata.title,
        "heading": chunk.metadata.heading,
        "chunk_index": chunk.metadata.chunk_index,
        "strategy": chunk.metadata.strategy,
    }
    for chunk in chunks
]

    texts = [build_search_text(chunk) for chunk in chunks]

    tokenizer = Tokenizer()
    corpus_tokens = tokenizer.tokenize(texts, return_as="tuple")

    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)

    target = index_path if index_path is not None else settings.sparse_index_path

    target.mkdir(parents=True, exist_ok=True)

    retriever.save(str(target))

    tokenizer.save_vocab(str(target))
    tokenizer.save_stopwords(str(target))
