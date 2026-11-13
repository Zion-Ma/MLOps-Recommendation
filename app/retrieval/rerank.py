from dataclasses import dataclass
from math import sqrt
from uuid import UUID

from sentence_transformers import SentenceTransformer

from app.retrieval.bm25 import BM25Result
from app.retrieval.dense import embed_texts


@dataclass(frozen=True)
class RerankedResult:
    article_id: UUID
    title: str
    url: str
    source: str
    category: str | None
    bm25_score: float
    dense_score: float
    rerank_score: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        left: First vector as a list of floats.
        right: Second vector as a list of floats.

    Returns:
        Cosine similarity score in the range [-1.0, 1.0]. If either vector has
        zero magnitude the function returns 0.0.

    Raises:
        ValueError: If the vectors are not the same length.
    """

    if len(left) != len(right):
        msg = "Vectors must have the same length"
        raise ValueError(msg)

    dot_product = sum(l_value * r_value for l_value, r_value in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def rerank_bm25_results_precomputed(
    query: str,
    bm25_results: list[BM25Result],
    candidate_vectors: dict[UUID, list[float]],
    article_texts: dict[UUID, str],
    model: SentenceTransformer,
    top_k: int = 10,
) -> list[RerankedResult]:
    """Rerank BM25 candidates against precomputed embeddings.

    Only the query is embedded per request; candidate vectors come from the
    vector store. Candidates missing a stored vector (e.g. articles ingested
    since the last indexing run) fall back to on-the-fly text encoding.
    """
    candidates = [
        result
        for result in bm25_results
        if result.article_id in candidate_vectors or result.article_id in article_texts
    ]
    if not candidates:
        return []

    query_vector = embed_texts(model, [query])[0]

    fallback = [result for result in candidates if result.article_id not in candidate_vectors]
    if fallback:
        fallback_texts = [article_texts[result.article_id] for result in fallback]
        fallback_vectors = embed_texts(model, fallback_texts)
        candidate_vectors = {
            **candidate_vectors,
            **{
                result.article_id: vector
                for result, vector in zip(fallback, fallback_vectors, strict=True)
            },
        }

    reranked_results = []
    for result in candidates:
        dense_score = cosine_similarity(query_vector, candidate_vectors[result.article_id])
        reranked_results.append(
            RerankedResult(
                article_id=result.article_id,
                title=result.title,
                url=result.url,
                source=result.source,
                category=result.category,
                bm25_score=result.score,
                dense_score=dense_score,
                rerank_score=dense_score,
            )
        )

    return sorted(
        reranked_results,
        key=lambda result: result.rerank_score,
        reverse=True,
    )[:top_k]


def rerank_bm25_results(
    query: str,
    bm25_results: list[BM25Result],
    article_texts: dict[UUID, str],
    model: SentenceTransformer,
    top_k: int = 10,
) -> list[RerankedResult]:
    """Rerank BM25 candidates using dense cosine similarity to the query."""
    candidates = [result for result in bm25_results if result.article_id in article_texts]
    if not candidates:
        return []

    query_vector = embed_texts(model, [query])[0]
    candidate_texts = [article_texts[result.article_id] for result in candidates]
    candidate_vectors = embed_texts(model, candidate_texts)

    reranked_results = []
    for result, candidate_vector in zip(candidates, candidate_vectors, strict=True):
        dense_score = cosine_similarity(query_vector, candidate_vector)

        reranked_results.append(
            RerankedResult(
                article_id=result.article_id,
                title=result.title,
                url=result.url,
                source=result.source,
                category=result.category,
                bm25_score=result.score,
                dense_score=dense_score,
                rerank_score=dense_score,
            )
        )

    return sorted(
        reranked_results,
        key=lambda result: result.rerank_score,
        reverse=True,
    )[:top_k]
