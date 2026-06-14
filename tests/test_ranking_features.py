from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.features.ranking import (
    CandidateScores,
    build_ranking_feature_row,
    compute_article_age_hours,
    compute_article_popularity,
    compute_user_category_affinity,
)


def test_compute_article_age_hours() -> None:
    now = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)
    published_at = now - timedelta(hours=6)

    assert compute_article_age_hours(published_at, now) == 6.0


def test_compute_article_age_hours_returns_none_without_publish_time() -> None:
    assert compute_article_age_hours(None) is None


def test_compute_article_popularity_counts_clicks_for_article() -> None:
    article_id = uuid4()
    other_article_id = uuid4()

    popularity = compute_article_popularity(
        article_id=article_id,
        clicked_article_ids=[article_id, other_article_id, article_id],
    )

    assert popularity == 2.0


def test_compute_user_category_affinity_uses_clicked_category_ratio() -> None:
    affinity = compute_user_category_affinity(
        candidate_category="sports",
        clicked_categories=["sports", "news", "sports", "health"],
    )

    assert affinity == 0.5


def test_compute_user_category_affinity_returns_zero_without_history() -> None:
    assert compute_user_category_affinity("sports", []) == 0.0
    assert compute_user_category_affinity(None, ["sports"]) == 0.0


def test_build_ranking_feature_row_combines_retrieval_and_user_features() -> None:
    user_id = uuid4()
    article_id = uuid4()
    other_article_id = uuid4()
    now = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)
    published_at = now - timedelta(hours=12)

    row = build_ranking_feature_row(
        group_id="impression-1",
        user_id=user_id,
        candidate=CandidateScores(
            article_id=article_id,
            bm25_score=3.5,
            dense_score=0.8,
            rerank_score=0.8,
        ),
        published_at=published_at,
        candidate_category="technology",
        clicked_article_ids=[article_id, other_article_id, article_id],
        clicked_categories=["technology", "news"],
        label=1,
        now=now,
    )

    assert row.user_id == user_id
    assert row.article_id == article_id
    assert row.group_id == "impression-1"
    assert row.bm25_score == 3.5
    assert row.dense_score == 0.8
    assert row.rerank_score == 0.8
    assert row.article_age_hours == 12.0
    assert row.article_popularity == 2.0
    assert row.user_category_affinity == 0.5
    assert row.label == 1
