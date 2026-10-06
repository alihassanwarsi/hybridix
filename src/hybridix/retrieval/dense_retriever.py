from hybridix.config import settings
from hybridix.indexing.dense_indexer import get_collection
from hybridix.indexing.embeddings import embed_query
from hybridix.models import Chunk, ChunkMetadata, RetrievalResult

def dense_search(query: str, top_k: int | None = None) -> list[RetrievalResult]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    top_k = top_k or settings.dense_top_k

    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        chunk = Chunk(
            content=content,
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                source=metadata.get("source"),
                file_type=metadata.get("file_type"),
                title=metadata.get("title"),
                heading=metadata.get("heading"),
                strategy=metadata.get("strategy"),
                chunk_index=metadata.get("chunk_index")
            )
        )

        retrieved.append(RetrievalResult(chunk=chunk, score=1.0 - distance))

    return retrieved