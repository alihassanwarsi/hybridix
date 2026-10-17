import json
from pathlib import Path
from datetime import datetime, timezone
from hybridix.config import settings
from hybridix.evaluation.dataset_loader import load_eval_cases
from hybridix.pipeline import answer_question

def run_answer_evaluation(dataset_path: Path, limit: int = 3):
    if limit < 1:
        raise ValueError("limit must be positive.")

    cases = load_eval_cases(dataset_path)

    selected_cases = cases[:limit]

    records = []

    for case in selected_cases:
        print(f"{case['id']}: {case['question']}")

        try:
            answer = answer_question(case["question"])

        except Exception as exc:
            records.append({
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "reference_sources": case["reference_sources"],
                "reference_answer": case["reference_answer"],
                "generated_answer": None,
                "refused": None,
                "should_refuse": case["should_refuse"],
                "citation_count": 0,
                "citations": [],
                "error": f"{type(exc).__name__}: {exc}"
            })

            print(f"Failed: {exc}")
            continue

        citations = []

        for citation in answer.citations:
            citations.append({
                "number": citation.number,
                "source": citation.source,
                "heading": citation.heading,
                "chunk_id": citation.chunk_id,
                "content": citation.content
            })

        record = {
            "id": case["id"],
            "question": case["question"],
            "category": case["category"],
            "reference_sources": case["reference_sources"],
            "reference_answer": case["reference_answer"],
            "generated_answer": answer.text,
            "refused": answer.refused,
            "should_refuse": case["should_refuse"],
            "citation_count": len(answer.citations),
            "citations": citations,
            "error": None
        }

        records.append(record)

        print(f"Answer: {answer.text}")
        print(f"Citations: {len(answer.citations)}")

    print(f"\nCollected {len(records)} evaluation records.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    report = {
        "timestamp": timestamp,
        "dataset": str(dataset_path),
        "generation_model": settings.generation_model,
        "total_questions": len(cases),
        "evaluated_questions": len(records),
        "results": records
    }

    report_path = Path(f"data/eval/reports/answer_results_{timestamp}.json")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Report saved: {report_path}")
