from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.recommendations import (
    get_cache_client,
    get_embedding_model,
    get_model_version,
    get_qdrant_client,
    get_ranker,
    get_search_client,
)
from app.db.models import Article, Recommendation, User, UserEvent
from app.db.session import get_db
from app.main import app
from app.retrieval.bm25 import BM25Result
from app.retrieval.rerank import RerankedResult

TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/news_rec"

engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class FakeRanker:
    """Scores candidates by their rerank score so ordering is predictable."""

    def predict(self, features):
        return [float(score) * 10 for score in features["rerank_score"]]


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_search_client] = lambda: object()
    app.dependency_overrides[get_embedding_model] = lambda: object()
    app.dependency_overrides[get_ranker] = FakeRanker
    app.dependency_overrides[get_model_version] = lambda: "test-model"
    app.dependency_overrides[get_cache_client] = lambda: None
    app.dependency_overrides[get_qdrant_client] = lambda: None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def make_article(db, title, category=None):
    article = Article(
        title=title,
        url=f"https://example.com/{uuid4()}",
        source="test",
        category=category,
        published_at=datetime.now(UTC),
        text=f"body of {title}",
    )
    db.add(article)
    db.commit()
    db.refresh(article)
    return article


def make_user(db, external_id=None):
    user = User(external_id=external_id or f"user-{uuid4()}")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def add_click(db, user, article):
    event = UserEvent(user_id=user.id, article_id=article.id, event_type="click")
    db.add(event)
    db.commit()


def bm25_result(article, score):
    return BM25Result(
        article_id=article.id,
        score=score,
        title=article.title,
        url=article.url,
        source=article.source,
        category=article.category,
    )


def reranked_result(article, bm25_score, rerank_score):
    return RerankedResult(
        article_id=article.id,
        title=article.title,
        url=article.url,
        source=article.source,
        category=article.category,
        bm25_score=bm25_score,
        dense_score=rerank_score,
        rerank_score=rerank_score,
    )


def test_recommendations_unknown_user_returns_404(client):
    response = client.get(f"/recommendations/{uuid4()}")

    assert response.status_code == 404


def test_recommendations_cold_start_falls_back_to_recent(client, db):
    user = make_user(db)
    make_article(db, "Fresh article")

    response = client.get(f"/recommendations/{user.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "recency_fallback"
    assert body["model_version"] == "test-model"
    assert len(body["items"]) >= 1


def test_recommendations_ranked_pipeline(client, db, monkeypatch):
    user = make_user(db)
    clicked = make_article(db, "Machine learning news", category="tech")
    low = make_article(db, "Low relevance", category="sports")
    high = make_article(db, "High relevance", category="tech")
    add_click(db, user, clicked)

    def fake_search(client_, query, top_k):
        assert "Machine learning news" in query
        return [bm25_result(low, 5.0), bm25_result(high, 4.0)]

    def fake_rerank(query, bm25_results, candidate_vectors, article_texts, model, top_k):
        return [
            reranked_result(low, bm25_score=5.0, rerank_score=0.1),
            reranked_result(high, bm25_score=4.0, rerank_score=0.9),
        ]

    monkeypatch.setattr("app.api.recommendations.search_bm25", fake_search)
    monkeypatch.setattr("app.api.recommendations.rerank_bm25_results_precomputed", fake_rerank)

    response = client.get(f"/recommendations/{user.external_id}?top_k=2")

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "ranked"
    titles = [item["title"] for item in body["items"]]
    assert titles == ["High relevance", "Low relevance"]
    assert [item["rank"] for item in body["items"]] == [1, 2]

    stored = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == user.id, Recommendation.model_version == "test-model")
        .all()
    )
    assert len(stored) == 2


def test_recommendations_excludes_clicked_articles(client, db, monkeypatch):
    user = make_user(db)
    clicked = make_article(db, "Clicked article", category="tech")
    other = make_article(db, "Other article", category="tech")
    add_click(db, user, clicked)

    def fake_search(client_, query, top_k):
        return [bm25_result(clicked, 9.0), bm25_result(other, 1.0)]

    def fake_rerank(query, bm25_results, candidate_vectors, article_texts, model, top_k):
        assert all(result.article_id != clicked.id for result in bm25_results)
        return [reranked_result(other, bm25_score=1.0, rerank_score=0.5)]

    monkeypatch.setattr("app.api.recommendations.search_bm25", fake_search)
    monkeypatch.setattr("app.api.recommendations.rerank_bm25_results_precomputed", fake_rerank)

    response = client.get(f"/recommendations/{user.id}")

    assert response.status_code == 200
    body = response.json()
    assert [item["title"] for item in body["items"]] == ["Other article"]


def test_recommendations_repeated_requests_upsert(client, db, monkeypatch):
    user = make_user(db)
    clicked = make_article(db, "Clicked article", category="tech")
    other = make_article(db, "Other article", category="tech")
    add_click(db, user, clicked)

    def fake_search(client_, query, top_k):
        return [bm25_result(other, 1.0)]

    def fake_rerank(query, bm25_results, candidate_vectors, article_texts, model, top_k):
        return [reranked_result(other, bm25_score=1.0, rerank_score=0.5)]

    monkeypatch.setattr("app.api.recommendations.search_bm25", fake_search)
    monkeypatch.setattr("app.api.recommendations.rerank_bm25_results_precomputed", fake_rerank)

    first = client.get(f"/recommendations/{user.id}")
    second = client.get(f"/recommendations/{user.id}")

    assert first.status_code == 200
    assert second.status_code == 200
    stored = (
        db.query(Recommendation)
        .filter(Recommendation.user_id == user.id, Recommendation.article_id == other.id)
        .all()
    )
    assert len(stored) == 1
