from functools import lru_cache
from sentence_transformers import SentenceTransformer
from hybridix.config import settings
from hybridix.models import Chunk

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)

def _validate_token_lengths(model: SentenceTransformer, texts: list[str]) -> None:
    max_tokens = int(model.max_seq_length)

    for index, text in enumerate(texts):
        token_count = len(model.tokenizer.encode(text, add_special_tokens = True))

        if token_count > max_tokens:
            raise ValueError(f"Embedding input {index} has {token_count} tokens, but the model allows only {max_tokens}. Split the text before embedding.")

def embed_chunks(chunks: list[Chunk]):
    model = get_embedding_model()
    texts = [chunk.content for chunk in chunks]

    _validate_token_lengths(model, texts)

    return model.encode(texts, normalize_embeddings=True)

def embed_query(query: str):
    model = get_embedding_model()

    _validate_token_lengths(model, [query])

    return model.encode(query, normalize_embeddings=True)

def embed_texts(texts: list[str]):
    model = get_embedding_model()

    _validate_token_lengths(model, texts)

    return model.encode(texts, normalize_embeddings=True)