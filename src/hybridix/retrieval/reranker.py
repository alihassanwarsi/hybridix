from functools import lru_cache
from sentence_transformers import CrossEncoder
from hybridix.models import RetrievalResult
from hybridix.config import settings
from hybridix.retrieval.hybrid_retriever import hybrid_search

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)

def rerank(query: str, results: list[RetrievalResult], top_k: int = 5,) -> list[RetrievalResult]:
    if not results:
        return []

    model = get_reranker()

    pairs = [
        (query, result.chunk.content)
        for result in results
    ]

    scores = model.predict(pairs)

    reranked = [
        RetrievalResult(chunk=result.chunk, score=float(score))
        for result, score in zip(results, scores)
    ]

    reranked.sort(key=lambda result: result.score, reverse=True)

    return reranked[:top_k]

def reranked_search(query: str, top_k: int = 5, candidate_k: int = 30,) -> list[RetrievalResult]:

    candidates = hybrid_search(query, top_k=candidate_k, candidate_k=candidate_k)

    return rerank(query=query, results=candidates, top_k=top_k)
