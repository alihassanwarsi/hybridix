import chromadb
from functools import lru_cache
from hybridix.config import settings
from hybridix.indexing.embeddings import embed_chunks
from hybridix.models import Chunk

@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=str(settings.dense_index_path))

def _get_or_create_collection(client):
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=None,
        configuration={
            "hnsw": {"space": "cosine"}
        }
    )

def get_collection():
    return _get_or_create_collection(get_client())

def reset_collection():
    client = get_client()

    try:
        client.delete_collection(name=settings.chroma_collection_name)
    except ValueError:
        pass

    return _get_or_create_collection(client)

def build_dense_index(chunks: list[Chunk]) -> None:
    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    collection = reset_collection()
    embeddings = embed_chunks(chunks)

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk.metadata.chunk_id)
        documents.append(chunk.content)

        metadata = {
            "source": chunk.metadata.source,
            "file_type": chunk.metadata.file_type,
            "strategy": chunk.metadata.strategy,
            "chunk_index": chunk.metadata.chunk_index,
        }

        if chunk.metadata.title is not None:
            metadata["title"] = chunk.metadata.title

        if chunk.metadata.heading is not None:
            metadata["heading"] = chunk.metadata.heading

        metadatas.append(metadata)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )