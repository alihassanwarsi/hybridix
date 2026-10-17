from uuid import uuid4
from pathlib import Path
from hybridix.config import settings
from hybridix.models import GeneratedAnswer
from hybridix.ingestion.loader import load_mdx
from hybridix.ingestion.chunking import chunk_by_headings
from hybridix.indexing.token_budget import fit_chunks_to_embedding_budget
from hybridix.indexing.version_builder import build_index_version
from hybridix.indexing.index_versions import activate_index_version
from hybridix.retrieval.reranker import reranked_search
from hybridix.generation.generator import generate_answer
from hybridix.generation.evidence_verifier import verify_answer

def build_indexes(source_root: Path, *, version: str | None = None, index_root: Path | None = None,) -> int:
    paths = sorted(source_root.rglob("*.mdx"))

    if not paths:
        raise ValueError(f"No MDX files found in {source_root}")

    chunks = []

    for path in paths:
        document = load_mdx(path, source_root)
        chunks.extend(chunk_by_headings(document))

    if not chunks:
        raise ValueError("No chunks generated.")

    chunks = fit_chunks_to_embedding_budget(chunks)

    if not chunks:
        raise ValueError("No chunks available for indexing.")

    version_name = version if version is not None else f"v_{uuid4().hex}"

    root = index_root if index_root is not None else settings.index_root

    build_index_version(chunks, version_name, index_root=root)

    activate_index_version(version_name, chunks, index_root=root)

    return len(chunks)

def answer_question(query: str, top_k: int = 5) -> GeneratedAnswer:
    if not query.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    results = reranked_search(query, top_k=top_k)

    if not results:
        return GeneratedAnswer(
            text="I don't have enough evidence to answer.",
            citations=[],
            refused=True,
            refusal_reason="no_results"
        )

    answer = generate_answer(query, results)
    answer.retrieved_evidence = results

    if answer.refused:
        answer.refusal_reason = "generator_refusal"
        return answer

    verification = verify_answer(answer)

    if not verification.supported:
        return GeneratedAnswer(
            text="I couldn't verify an answer from the available evidence.",
            citations=[],
            refused=True,
            refusal_reason="verifier_rejection",
            verification_reason=verification.reason,
            unverified_answer=answer.text,
            retrieved_evidence=results
        )

    return answer