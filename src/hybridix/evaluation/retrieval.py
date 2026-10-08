from hybridix.models import RetrievalResult

def is_relevant(result: RetrievalResult, relevant: list[dict]) -> bool:
    for target in relevant:
        source_matches = result.chunk.metadata.source == target["source"]

        heading = target.get("heading")

        heading_matches = heading is None or result.chunk.metadata.heading == heading

        if source_matches and heading_matches:
            return True

    return False

def hit_at_k(results: list[RetrievalResult], relevant: list[dict], k: int) -> float:
    for result in results[:k]:
        if is_relevant(result, relevant):
            return 1.0

    return 0.0

def reciprocal_rank(results: list[RetrievalResult], relevant: list[dict], k: int) -> float:
    for rank, result in enumerate(results[:k], start=1):
        if is_relevant(result, relevant):
            return 1.0 / rank

    return 0.0