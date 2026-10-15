from pathlib import Path
from hybridix.models import GeneratedAnswer
from hybridix.ingestion.loader import load_mdx
from hybridix.ingestion.chunking import chunk_by_headings
from hybridix.indexing.token_budget import fit_chunks_to_embedding_budget
from hybridix.indexing.dense_indexer import build_dense_index
from hybridix.indexing.sparse_indexer import build_sparse_index
from hybridix.retrieval.reranker import reranked_search
from hybridix.generation.generator import generate_answer
from hybridix.generation.evidence_verifier import verify_answer

def build_indexes(source_root: Path) -> int:
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
    
    build_dense_index(chunks)
    build_sparse_index(chunks)

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
            refused=True
        )

    answer = generate_answer(query, results)

    verification = verify_answer(answer)

    if not verification.supported:
        return GeneratedAnswer(
            text="I couldn't verify an answer from the available evidence.",
            citations=[],
            refused=True
        )

    return answer