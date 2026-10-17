from functools import lru_cache
from sentence_transformers import CrossEncoder
from hybridix.models import RetrievalResult
from hybridix.config import settings
from hybridix.retrieval.dense_retriever import dense_search
from hybridix.retrieval.sparse_retriever import sparse_search
from hybridix.indexing.index_versions import get_active_index_paths

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)

def rerank(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    if not results:
        return []

    model = get_reranker()

    pairs = [(query, result.chunk.content) for result in results]

    scores = model.predict(pairs)

    reranked = [
        RetrievalResult(chunk=result.chunk, score=float(score))
        for result, score in zip(results, scores)
    ]

    reranked.sort(key=lambda result: result.score, reverse=True)

    return reranked[:top_k]

def reranked_search(query: str, top_k: int = 5, candidate_k: int = 30) -> list[RetrievalResult]:
    dense_path, sparse_path = get_active_index_paths()

    dense_results = dense_search(query, top_k=candidate_k, index_path=dense_path)

    sparse_results = sparse_search(query, top_k=candidate_k, index_path=sparse_path)

    candidates: dict[str, RetrievalResult] = {}

    for result in dense_results + sparse_results:
        chunk_id = result.chunk.metadata.chunk_id
        candidates[chunk_id] = result

    return rerank(query=query, results=list(candidates.values()), top_k=top_k)