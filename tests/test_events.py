from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Article, User, UserEvent
from app.db.session import get_db
from app.main import app

TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/news_rec"

engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def make_article(db):
    article = Article(
        title="Event article",
        url=f"https://example.com/{uuid4()}",
        source="test",
        published_at=datetime.now(UTC),
    )
    db.add(article)
    db.commit()
    db.refresh(article)
    return article


def make_user(db):
    user = User(external_id=f"user-{uuid4()}")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_record_click_event(client, db):
    user = make_user(db)
    article = make_article(db)

    response = client.post(
        "/events/click",
        json={
            "user_id": str(user.id),
            "article_id": str(article.id),
            "model_version": "test-model",
            "position": 3,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["event_type"] == "click"
    assert body["user_id"] == str(user.id)

    stored = db.get(UserEvent, body["event_id"])
    assert stored is not None
    assert stored.position == 3
    assert stored.model_version == "test-model"


def test_record_impression_with_metadata(client, db):
    user = make_user(db)
    article = make_article(db)

    response = client.post(
        "/events/impression",
        json={
            "user_id": user.external_id,
            "article_id": str(article.id),
            "metadata": {"clicked": 0, "slot": "homepage"},
        },
    )

    assert response.status_code == 201
    stored = db.get(UserEvent, response.json()["event_id"])
    assert stored.event_metadata == {"clicked": 0, "slot": "homepage"}


def test_record_dwell_creates_unknown_external_user(client, db):
    article = make_article(db)
    external_id = f"new-user-{uuid4()}"

    response = client.post(
        "/events/dwell",
        json={
            "user_id": external_id,
            "article_id": str(article.id),
            "metadata": {"dwell_seconds": 42},
        },
    )

    assert response.status_code == 201
    from app.db.repositories import UserRepository

    user = UserRepository(db).get_by_external_id(external_id)
    assert user is not None


def test_record_event_missing_article_returns_404(client, db):
    user = make_user(db)

    response = client.post(
        "/events/click",
        json={"user_id": str(user.id), "article_id": str(uuid4())},
    )

    assert response.status_code == 404


def test_record_event_unknown_user_uuid_returns_404(client, db):
    article = make_article(db)

    response = client.post(
        "/events/skip",
        json={"user_id": str(uuid4()), "article_id": str(article.id)},
    )

    assert response.status_code == 404


def test_metrics_endpoint_exposes_prometheus_metrics(client):
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
