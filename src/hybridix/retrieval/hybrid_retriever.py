from hybridix.models import RetrievalResult
from hybridix.retrieval.dense_retriever import dense_search
from hybridix.retrieval.sparse_retriever import sparse_search

def rrf_score(rank: int, k: int = 60) -> float:
    if rank < 1:
        raise ValueError("Rank must be 1 or greater.")
    return 1 / (k + rank)

def hybrid_search(query: str, top_k: int = 10, rrf_k: int = 60,) -> list[RetrievalResult]:

    dense_results = dense_search(query)
    sparse_results = sparse_search(query)

    rrf_scores = {}
    chunks = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk.metadata.chunk_id

        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score(rank, rrf_k)
        chunks[chunk_id] = result.chunk

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result.chunk.metadata.chunk_id

        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score(rank, rrf_k)
        chunks[chunk_id] = result.chunk

    ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    return [
        RetrievalResult(chunk=chunks[chunk_id], score=score) 
        for chunk_id, score in ranked[:top_k]
    ]