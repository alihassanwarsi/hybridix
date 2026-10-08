import json
from collections.abc import Callable
from pathlib import Path
from hybridix.evaluation.retrieval import hit_at_k, reciprocal_rank
from hybridix.models import RetrievalResult
from hybridix.retrieval.dense_retriever import dense_search
from hybridix.retrieval.hybrid_retriever import hybrid_search
from hybridix.retrieval.sparse_retriever import sparse_search

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

def evaluate_retriever(cases: list[dict], retriever: Retriever, k: int = 5) -> dict[str, float]:
    hits = []
    reciprocal_ranks = []

    for case in cases:
        results = retriever(case["query"], k)

        hits.append(hit_at_k(results, case["relevant"], k))

        reciprocal_ranks.append(reciprocal_rank(results, case["relevant"], k))

    return {
        f"hit@{k}": sum(hits) / len(hits),
        f"mrr@{k}": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }

def run_retrieval_evaluation(path: Path, k: int = 5,) -> None:
    cases = load_eval_cases(path)

    retrievers = {
        "dense": dense_search,
        "sparse": sparse_search,
        "hybrid": hybrid_search
    }

    for name, retriever in retrievers.items():
        metrics = evaluate_retriever(cases, retriever, k)

        print(
            name,
            f"Hit@{k}: {metrics[f'hit@{k}']:.3f}",
            f"MRR@{k}: {metrics[f'mrr@{k}']:.3f}",
        )