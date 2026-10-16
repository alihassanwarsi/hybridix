import argparse
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hybridix",
        description="Hybridix, a modular RAG engine"
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True
    )

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Evaluate retrieval performance"
    )

    evaluate_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/retrieval.jsonl")
    )

    evaluate_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to evaluate per question"
    )

    ask_parser = commands.add_parser(
        "ask",
        help="Ask a question about indexed documents"
    )

    ask_parser.add_argument(
        "question",
        help="Question to ask Hybridix"
    )

    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence chunks to retrieve"
    )

    ask_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show additional debugging information"
    )

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        if args.top_k < 1:
            parser.error("--top-k must be positive")

        from hybridix.evaluation.runner import run_retrieval_evaluation

        run_retrieval_evaluation(path=args.dataset, k=args.top_k)

    if args.command == "ask":
        if args.top_k < 1:
            parser.error("--top-k must be positive")

        from hybridix.pipeline import answer_question

        answer = answer_question(
            query=args.question,
            top_k=args.top_k
        )

        print(f"\nAnswer:\n{answer.text}")

        if answer.citations:
            print("\nSources:")

            for citation in answer.citations:
                print(
                    f"[{citation.number}] {citation.source}"
                    f" | {citation.heading or 'No heading'}"
                )

        if args.debug:
            print("\nDebug:")
            print(f"Refused: {answer.refused}")
            print(f"Refusal reason: {answer.refusal_reason or 'none'}")

            if answer.verification_reason is not None:
                print(f"Verifier reason: {answer.verification_reason}")

            if answer.unverified_answer is not None:
                print(f"\nUnverified answer:\n{answer.unverified_answer}")

            if answer.retrieved_evidence:
                print("\nRetrieved evidence:")

                for rank, result in enumerate(answer.retrieved_evidence, start=1):
                    metadata = result.chunk.metadata

                    print(
                        f"{rank}. {metadata.source}"
                        f" | {metadata.heading or 'No heading'}"
                    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
