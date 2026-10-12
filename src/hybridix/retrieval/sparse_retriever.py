import bm25s
from bm25s.tokenization import Tokenizer
from hybridix.config import settings
from hybridix.models import Chunk, ChunkMetadata, RetrievalResult


def sparse_search(query: str, top_k: int | None = None) -> list[RetrievalResult]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    top_k = top_k or settings.sparse_top_k
    index_path = settings.sparse_index_path

    if not (index_path / "params.index.json").is_file():
        raise FileNotFoundError(f"Sparse index not found at {index_path}. Build the index before searching.")

    retriever = bm25s.BM25.load(
        str(index_path),
        load_corpus=True
    )

    available_chunks = len(retriever.corpus)
    top_k = min(top_k, available_chunks)

    tokenizer = Tokenizer()
    tokenizer.load_vocab(str(settings.sparse_index_path))
    tokenizer.load_stopwords(str(settings.sparse_index_path))

    query_tokens = tokenizer.tokenize([query], update_vocab=False)

    results, scores = retriever.retrieve(query_tokens, k=top_k)

    retrieved = []

    for document, score in zip(results[0], scores[0]):
        chunk = Chunk(
            content=document["text"],
            metadata=ChunkMetadata(
                chunk_id=document["id"],
                source=document["source"],
                file_type=document["file_type"],
                title=document.get("title"),
                strategy=document["strategy"],
                chunk_index=document["chunk_index"],
                heading=document.get("heading")
            )
        )

        retrieved.append(RetrievalResult(chunk=chunk, score=float(score)))

    return retrieved