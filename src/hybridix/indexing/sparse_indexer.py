import bm25s
from bm25s.tokenization import Tokenizer
from hybridix.config import settings
from hybridix.models import Chunk
from hybridix.indexing.search_text import build_search_text

def build_sparse_index(chunks: list[Chunk]) -> None:
    if not chunks:
        raise ValueError("No chunks provided for indexing.")

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

    index_path = settings.sparse_index_path
    index_path.mkdir(parents=True, exist_ok=True)

    retriever.save(str(index_path))

    tokenizer.save_vocab(str(index_path))
    tokenizer.save_stopwords(str(index_path))