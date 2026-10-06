from functools import lru_cache
from sentence_transformers import SentenceTransformer
from hybridix.models import Chunk

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(chunks: list[Chunk]):
    model = get_embedding_model()
    texts = [chunk.content for chunk in chunks]

    return model.encode(texts, normalize_embeddings=True)

def embed_query(query: str):
    model = get_embedding_model()
    return model.encode(query, normalize_embeddings=True)