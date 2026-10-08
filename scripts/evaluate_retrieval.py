from pathlib import Path
from hybridix.evaluation.runner import run_retrieval_evaluation

run_retrieval_evaluation(
    Path("data/eval/retrieval.jsonl"),
    k=5
)