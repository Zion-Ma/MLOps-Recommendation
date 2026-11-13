from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Article, User, UserEvent
from pipelines.compute_ctr import compute_ctr

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


def test_compute_ctr_counts_recent_events(db):
    user = User(external_id=f"user-{uuid4()}")
    article = Article(title="CTR article", url=f"https://example.com/{uuid4()}", source="test")
    db.add_all([user, article])
    db.commit()

    for _ in range(4):
        db.add(UserEvent(user_id=user.id, article_id=article.id, event_type="impression"))
    db.add(UserEvent(user_id=user.id, article_id=article.id, event_type="click"))
    db.commit()

    report = compute_ctr(db, window_hours=1)

    assert report["impressions"] >= 4
    assert report["clicks"] >= 1
    assert report["ctr"] is not None
    # Clicks are not tied to in-window impressions, so on a shared database
    # the ratio can exceed 1; only positivity is guaranteed.
    assert report["ctr"] > 0.0
