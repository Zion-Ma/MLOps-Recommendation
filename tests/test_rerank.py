from uuid import uuid4

import pytest

from app.retrieval.bm25 import BM25Result
from app.retrieval.rerank import cosine_similarity, rerank_bm25_results


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

def test_cosine_similarity_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])

def fake_embed_texts(model, texts):
    mapping = {
        "query": [1.0, 0.0],
        "bad candidate": [0.0, 1.0],
        "good candidate": [1.0, 0.0],
    }
    return [mapping[text] for text in texts]

def test_rerank_bm25_results_orders_by_dense_score(monkeypatch) -> None:
    good_id = uuid4()
    bad_id = uuid4()

    def fake_embed_texts(model, texts):
        mapping = {
            "query": [1.0, 0.0],
            "bad candidate": [0.0, 1.0],
            "good candidate": [1.0, 0.0],
        }
        return [mapping[text] for text in texts]

    monkeypatch.setattr("app.retrieval.rerank.embed_texts", fake_embed_texts)

    results = rerank_bm25_results(
        query="query",
        bm25_results=[
            BM25Result(
                article_id=bad_id,
                score=100.0,
                title="Bad",
                url="https://example.com/bad",
                source="test",
                category=None,
            ),
            BM25Result(
                article_id=good_id,
                score=1.0,
                title="Good",
                url="https://example.com/good",
                source="test",
                category=None,
            ),
        ],
        article_texts={
            bad_id: "bad candidate",
            good_id: "good candidate",
        },
        model=object(),
        top_k=2,
    )

    assert [result.article_id for result in results] == [good_id, bad_id]
    assert results[0].dense_score == 1.0
    assert results[0].rerank_score == 1.0
    assert results[1].bm25_score == 100.0