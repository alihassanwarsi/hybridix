import json
from pathlib import Path

def load_eval_cases(path: Path) -> list[dict]:
    cases = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(json.loads(line))

    if not cases:
        raise ValueError("No evaluation cases found.")

    return cases
