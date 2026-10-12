
import pytest
from hybridix.indexing import token_budget, embeddings
from hybridix.indexing.search_text import build_search_text
from hybridix.models import Chunk, ChunkMetadata

class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [0] * (len(text) + 2)

class FakeModel:
    max_seq_length = 80
    tokenizer = FakeTokenizer()

    def encode(self, texts, normalize_embeddings=True):
        return [1, 2, 3]

def make_chunk(content, title="Gateway"):
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            chunk_id="original-id",
            source="gateway.mdx",
            file_type="mdx",
            title=title,
            heading="Connecting",
            strategy="heading",
            chunk_index=0
        )
    )

def test_long_chunk_is_split_without_losing_text(monkeypatch):
    monkeypatch.setattr(
        token_budget,
        "get_embedding_model",
        lambda: FakeModel(),
    )

    original = make_chunk("Discord gateway connections and sharding. " * 20)

    chunks = token_budget.fit_chunks_to_embedding_budget([original])

    assert len(chunks) > 1
    assert "".join(chunk.content for chunk in chunks) == original.content

    for chunk in chunks:
        text = build_search_text(chunk)
        assert len(FakeTokenizer().encode(text)) <= 80

    ids = [chunk.metadata.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_split_ids_are_deterministic(monkeypatch):
    monkeypatch.setattr(
        token_budget,
        "get_embedding_model",
        lambda: FakeModel()
    )

    chunk = make_chunk("Very long documentation. " * 30)

    first = token_budget.fit_chunks_to_embedding_budget([chunk])
    second = token_budget.fit_chunks_to_embedding_budget([chunk])

    assert [c.metadata.chunk_id for c in first] == [
        c.metadata.chunk_id for c in second
    ]

def test_metadata_too_large_is_rejected(monkeypatch):
    monkeypatch.setattr(
        token_budget,
        "get_embedding_model",
        lambda: FakeModel()
    )

    chunk = make_chunk(content="Some content", title="Very long title " * 15)

    with pytest.raises(ValueError, match="Not enough embedding tokens"):
        token_budget.fit_chunks_to_embedding_budget([chunk])

def test_direct_embedding_rejects_truncation(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: FakeModel()
    )

    with pytest.raises(ValueError, match="Split the text"):
        embeddings.embed_texts(["X" * 200])
