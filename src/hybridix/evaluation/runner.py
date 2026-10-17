import json
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Callable
from hybridix.config import settings
from hybridix.evaluation.retrieval import hit_at_k, reciprocal_rank
from hybridix.models import RetrievalResult
from hybridix.retrieval.dense_retriever import dense_search
from hybridix.retrieval.hybrid_retriever import hybrid_search
from hybridix.retrieval.sparse_retriever import sparse_search
from hybridix.retrieval.reranker import reranked_search
from hybridix.evaluation.dataset_loader import load_eval_cases
from hybridix.indexing.index_versions import get_active_index_paths

Retriever = Callable[[str, int], list[RetrievalResult]]

def evaluate_retriever(cases: list[dict], retriever: Retriever, k: int = 5, show_failures: bool = False, details: list[dict] | None = None) -> dict[str, float]:
    hits = []
    reciprocal_ranks = []

    for number, case in enumerate(cases, start=1):
        results = retriever(case["query"], k)

        hit = hit_at_k(results, case["relevant"], k)
        rr = reciprocal_rank(results, case["relevant"], k)

        hits.append(hit)
        reciprocal_ranks.append(rr)

        if details is not None:
            details.append({
                "question_number": number,
                "query": case["query"],
                "expected": case["relevant"],
                "hit": bool(hit),
                "reciprocal_rank": float(rr),
                "retrieved": [
                    {
                        "rank": rank,
                        "source": result.chunk.metadata.source,
                        "heading": result.chunk.metadata.heading,
                        "chunk_id": result.chunk.metadata.chunk_id,
                        "score": float(result.score)
                    }
                    for rank, result in enumerate(results[:k], start=1)
                ]
            })

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


def run_retrieval_evaluation(path: Path, k: int = 5) -> None:
    cases = load_eval_cases(path)

    dense_path, sparse_path = get_active_index_paths()

    retrievers = {
    "dense": lambda query, k: dense_search(query, top_k=k, index_path=dense_path),
    "sparse": lambda query, k: sparse_search(query, top_k=k, index_path=sparse_path),
    "hybrid": hybrid_search,
    "reranked": reranked_search,
}

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(path),
        "total_questions": len(cases),
        "k": k,
        "config": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "dense_index_path": str(dense_path),
            "sparse_index_path": str(sparse_path)
        },
        "retrievers": {}
    }

    for name, retriever in retrievers.items():
        details = []

        metrics = evaluate_retriever(cases, retriever, k, show_failures=True, details=details)

        report["retrievers"][name] = {"metrics": metrics, "questions": details}

        print(
            name,
            f"Hit@{k}: {metrics[f'hit@{k}']:.3f}",
            f"MRR@{k}: {metrics[f'mrr@{k}']:.3f}"
        )

    report_dir = Path("data/eval/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    report_path = report_dir / f"retrieval_{timestamp}.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nEvaluation report saved: {report_path}")
