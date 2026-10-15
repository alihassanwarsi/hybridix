import re
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import FactualCorrectness, Faithfulness, AnswerRelevancy
from hybridix.config import settings
from hybridix.evaluation.dataset_loader import load_eval_cases

REFUSAL_MESSAGES = {
    "I don't have enough evidence to answer.",
    "I couldn't verify an answer from the available evidence."
}

def was_refused(case: dict) -> bool:
    if case.get("refused") is not None:
        return case["refused"]

    return case["generated_answer"].strip() in REFUSAL_MESSAGES

def save_results(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temporary.replace(path)

def calculate_score(metric, **inputs):
    try:
        score = float(metric.score(**inputs).value)

        if not math.isfinite(score):
            return None, "Metric returned an invalid score."
        return score, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

def run_ragas_evaluation(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))

    golden = {
        case["id"]: case
        for case in load_eval_cases(Path("data/eval/golden_dataset.jsonl"))
    }

    if settings.openai_api_key is None:
        raise ValueError("OpenAI API key is not configured.")

    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    llm = llm_factory(settings.evaluation_model, provider="openai", client=client)

    factual = FactualCorrectness(llm=llm, mode="f1")
    faithful = Faithfulness(llm=llm)

    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    output_path = report_path.with_name(f"ragas_results_{timestamp}.json")

    output = {
        "source_report": str(report_path),
        "evaluation_model": settings.evaluation_model,
        "results": [],
        "summary": {}
    }

    for case in report["results"]:
        case_id = case["id"]

        row = {
            "id": case_id,
            "factual_correctness": None,
            "faithfulness": None,
            "answer_relevancy": None,
            "errors": [],
            "refusal_correct": None,
            "citations_valid": None
        }

        output["results"].append(row)

        if case.get("error") or not case.get("generated_answer"):
            row["errors"].append("Answer generation failed.")

        elif case_id not in golden:
            row["errors"].append("Question not found in golden dataset.")

        elif golden[case_id]["should_refuse"]:
            row["refusal_correct"] = was_refused(case)
            row["status"] = "refusal_checked"

        elif was_refused(case):
            row["refusal_correct"] = False
            row["status"] = "unexpected_refusal"

        else:
            row["refusal_correct"] = True
            print(f"Evaluating {case_id}...")

            cited_numbers = {int(number) for number in re.findall(r"\[(\d+)\]", case["generated_answer"])}
            citations = case.get("citations", [])

            available_numbers = {citation["number"] for citation in citations}

            row["citations_valid"] = (
                bool(cited_numbers)
                and cited_numbers.issubset(available_numbers)
                and all(citation.get("content") for citation in citations)
            )

            print("  Citations Valid:", row["citations_valid"])

            score, error = calculate_score(
                factual,
                response=case["generated_answer"],
                reference=golden[case_id]["reference_answer"]
            )

            row["factual_correctness"] = score

            if error:
                row["errors"].append(f"Factual Correctness: {error}")

            save_results(output_path, output)

            contexts = [
                citation["content"]
                for citation in case.get("citations", [])
                if citation.get("content")
            ]

            if contexts:
                score, error = calculate_score(
                    faithful,
                    user_input=case["question"],
                    response=case["generated_answer"],
                    retrieved_contexts=contexts
                )

                row["faithfulness"] = score

                if error:
                    row["errors"].append(f"Faithfulness: {error}")
            else:
                row["status"] = "faithfulness_skipped_no_evidence"

            score, error = calculate_score(
                relevancy,
                user_input=case["question"],
                response=case["generated_answer"]
            )

            row["answer_relevancy"] = score

            if error:
                row["errors"].append(f"Answer Relevancy: {error}")

            print("  Factual Correctness:", row["factual_correctness"])
            print("  Faithfulness:", row["faithfulness"])
            print("  Answer Relevancy:", row["answer_relevancy"])

        save_results(output_path, output)

    for metric in ("factual_correctness", "faithfulness", "answer_relevancy"):
        values = [
            row[metric]
            for row in output["results"]
            if row[metric] is not None
        ]

        output["summary"][metric] = {
            "average": round(sum(values) / len(values), 3) if values else None,
            "scored": len(values)
        }

    for field, name in (
        ("refusal_correct", "refusal_accuracy"),
        ("citations_valid", "citation_validity")
    ):
        checked = [
            row[field]
            for row in output["results"]
            if row[field] is not None
        ]

        output["summary"][name] = {
            "rate": round(sum(checked) / len(checked), 3) if checked else None,
            "checked": len(checked)
        }

    output["summary"]["failed_cases"] = sum(bool(row["errors"]) for row in output["results"])

    save_results(output_path, output)

    print("\nSummary:", output["summary"])
    print("Report saved:", output_path)
