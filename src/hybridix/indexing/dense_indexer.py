import chromadb
from chromadb.errors import NotFoundError
from pathlib import Path
from functools import lru_cache
from hybridix.config import settings
from hybridix.indexing.embeddings import embed_texts
from hybridix.indexing.search_text import build_search_text
from hybridix.models import Chunk

@lru_cache(maxsize=8)
def _cached_client(path: str):
    return chromadb.PersistentClient(path=path)

def get_client(index_path: Path | None = None):
    path = index_path if index_path is not None else settings.dense_index_path
    return _cached_client(str(path.resolve()))

def _get_or_create_collection(client):
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=None,
        configuration={
            "hnsw": {"space": "cosine"}
        }
    )

def get_collection(index_path: Path | None = None):
    if index_path is None:
        client = get_client()
    else:
        client = get_client(index_path)

    return _get_or_create_collection(client)

def reset_collection():
    client = get_client()

    try:
        client.delete_collection(name=settings.chroma_collection_name)
    except NotFoundError:
        pass

    return _get_or_create_collection(client)

def build_dense_index(chunks: list[Chunk], *, index_path: Path | None = None) -> None:
    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    if index_path is not None and index_path.exists():
        raise FileExistsError(f"Index destination already exists: {index_path}")

    search_texts = [build_search_text(chunk) for chunk in chunks]

    embeddings = embed_texts(search_texts)

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

    if index_path is None:
        collection = reset_collection()
    else:
        collection = get_collection(index_path)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )