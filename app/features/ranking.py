from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID


@dataclass(frozen=True)
class CandidateScores:
    article_id: UUID
    bm25_score: float
    dense_score: float
    rerank_score: float

@dataclass(frozen=True)
class RankingFeatureRow:
    group_id: str
    user_id: UUID
    article_id: UUID
    bm25_score: float
    dense_score: float
    rerank_score: float
    article_age_hours: float | None
    article_popularity: float
    user_category_affinity: float
    label: int | None = None

# compute article age in hours
def compute_article_age_hours(
    published_at: datetime | None,
    now: datetime | None = None,
) -> float | None:
    if published_at is None:
        return None

    if now is None:
        now = datetime.now(UTC)

    age = now - published_at
    return age.total_seconds() / 3600

# compute article popularity based on click history
def compute_article_popularity(article_id: UUID, clicked_article_ids: list[UUID]) -> float:
    counter = clicked_article_ids.count(article_id)
    return float(counter)

# compute user category affinity
def compute_user_category_affinity(
    candidate_category: str | None,
    clicked_categories: list[str],
) -> float:
    if candidate_category is None or not clicked_categories:
        return 0.0

    matches = clicked_categories.count(candidate_category)
    return matches / len(clicked_categories)


def build_ranking_feature_row(
    group_id: str,
    user_id: UUID,
    candidate: CandidateScores,
    published_at: datetime | None,
    candidate_category: str | None,
    clicked_article_ids: list[UUID],
    clicked_categories: list[str],
    label: int | None = None,
    now: datetime | None = None,
) -> RankingFeatureRow:
    article_age_hours = compute_article_age_hours(published_at, now)
    article_popularity = compute_article_popularity(candidate.article_id, clicked_article_ids)
    user_category_affinity = compute_user_category_affinity(candidate_category, clicked_categories)

    return RankingFeatureRow(
        group_id=group_id,
        user_id=user_id,
        article_id=candidate.article_id,
        bm25_score=candidate.bm25_score,
        dense_score=candidate.dense_score,
        rerank_score=candidate.rerank_score,
        article_age_hours=article_age_hours,
        article_popularity=article_popularity,
        user_category_affinity=user_category_affinity,
        label=label,
    )
