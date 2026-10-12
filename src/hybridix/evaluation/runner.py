import json
from pathlib import Path
from collections.abc import Callable
from hybridix.evaluation.retrieval import hit_at_k, reciprocal_rank
from hybridix.models import RetrievalResult
from hybridix.retrieval.dense_retriever import dense_search
from hybridix.retrieval.hybrid_retriever import hybrid_search
from hybridix.retrieval.sparse_retriever import sparse_search
from hybridix.retrieval.reranker import reranked_search

Retriever = Callable[[str, int], list[RetrievalResult]]

def load_eval_cases(path: Path) -> list[dict]:
    cases = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(json.loads(line))

    if not cases:
        raise ValueError("No evaluation cases found.")

    return cases

def evaluate_retriever(cases: list[dict], retriever: Retriever, k: int = 5, show_failures: bool = False) -> dict[str, float]:
    hits = []
    reciprocal_ranks = []


    for number, case in enumerate(cases, start=1):
        results = retriever(case["query"], k)

        hit = hit_at_k(results, case["relevant"], k)
        rr = reciprocal_rank(results, case["relevant"], k)

        hits.append(hit)
        reciprocal_ranks.append(rr)

        if show_failures and hit == 0:
            print(f"\n  MISS Q{number}: {case['query']}")
            print(f"  Expected: {case['relevant']}")

            for rank, result in enumerate(results[:3], start=1):
                meta = result.chunk.metadata
                print(
                        f"    {rank}. {meta.source}"
                        f" | {meta.heading or 'No heading'}"
                    )

    return {
        f"hit@{k}": sum(hits) / len(hits),
        f"mrr@{k}": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }

def run_retrieval_evaluation(path: Path, k: int = 5,) -> None:
    cases = load_eval_cases(path)

    retrievers = {
        "dense": dense_search,
        "sparse": sparse_search,
        "hybrid": hybrid_search,
        "reranked": reranked_search
    }

    for name, retriever in retrievers.items():
        metrics = evaluate_retriever(cases, retriever, k,show_failures=True)

        print(
            name,
            f"Hit@{k}: {metrics[f'hit@{k}']:.3f}",
            f"MRR@{k}: {metrics[f'mrr@{k}']:.3f}",
        )