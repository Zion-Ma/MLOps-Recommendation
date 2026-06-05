from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Article, ModelVersion, User, UserEvent
from app.db.repositories import (
    ArticleRepository,
    ModelVersionRepository,
    UserEventRepository,
    UserRepository,
)

TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/news_rec"

engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db() -> Session:
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_article_repository_create_and_get_by_url(db: Session) -> None:
    repo = ArticleRepository(db)
    url = f"https://example.com/{uuid4()}"

    article = repo.create(
        Article(
            title="Test Article",
            url=url,
            source="test",
        )
    )

    found = repo.get_by_url(url)

    assert found is not None
    assert found.id == article.id
    assert found.title == "Test Article"


def test_user_repository_create_and_get_by_external_id(db: Session) -> None:
    repo = UserRepository(db)
    external_id = f"user-{uuid4()}"

    user = repo.create(User(external_id=external_id))

    found = repo.get_by_external_id(external_id)

    assert found is not None
    assert found.id == user.id


def test_user_event_repository_create_and_list_for_user(db: Session) -> None:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    event_repo = UserEventRepository(db)

    article = article_repo.create(
        Article(
            title="Event Article",
            url=f"https://example.com/{uuid4()}",
            source="test",
        )
    )
    user = user_repo.create(User(external_id=f"user-{uuid4()}"))

    event = event_repo.create(
        UserEvent(
            user_id=user.id,
            article_id=article.id,
            event_type="click",
            position=1,
            model_version="test-model",
            event_metadata={"source": "test"},
        )
    )

    events = event_repo.list_for_user(user.id)

    assert len(events) >= 1
    assert events[0].id == event.id
    assert events[0].event_type == "click"


def test_model_version_repository_create_and_get_by_version(db: Session) -> None:
    repo = ModelVersionRepository(db)
    version = f"model-{uuid4()}"

    model_version = repo.create(
        ModelVersion(
            name="ranker",
            version=version,
            status="staging",
            metrics={"ndcg_at_10": 0.5},
        )
    )

    found = repo.get_by_version(version)

    assert found is not None
    assert found.id == model_version.id
    assert found.metrics == {"ndcg_at_10": 0.5}
