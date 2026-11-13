from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Article, ModelVersion, Recommendation, User, UserEvent


class ArticleRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, article: Article) -> Article:
        self.db.add(article)
        self.db.commit()
        self.db.refresh(article)
        return article

    def get_by_url(self, url: str) -> Article | None:
        statement = select(Article).where(Article.url == url)
        return self.db.scalar(statement)

    def list_recent(self, limit: int = 100) -> list[Article]:
        statement = select(Article).order_by(Article.created_at.desc()).limit(limit)
        return list(self.db.scalars(statement).all())

    def get_by_source_external_id(self, source: str, external_id: str) -> Article | None:
        statement = select(Article).where(
            Article.source == source,
            Article.external_id == external_id,
        )
        return self.db.scalar(statement)

    def get_by_ids(self, article_ids: list[UUID]) -> list[Article]:
        if not article_ids:
            return []

        statement = select(Article).where(Article.id.in_(article_ids))
        return list(self.db.scalars(statement).all())


class UserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, user: User) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_by_external_id(self, external_id: str) -> User | None:
        statement = select(User).where(User.external_id == external_id)
        return self.db.scalar(statement)


class UserEventRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, event: UserEvent) -> UserEvent:
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def list_for_user(self, user_id: UUID, limit: int = 100) -> list[UserEvent]:
        statement = (
            select(UserEvent)
            .where(UserEvent.user_id == user_id)
            .order_by(UserEvent.timestamp.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())

    def list_labeled_impressions(self, limit: int = 1000) -> list[UserEvent]:
        statement = (
            select(UserEvent)
            .where(UserEvent.event_type == "impression")
            .where(UserEvent.event_metadata["clicked"].as_integer().is_not(None))
            .order_by(UserEvent.timestamp.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())

    def count_clicks_by_article(self, article_ids: list[UUID]) -> dict[UUID, int]:
        """Global click counts per article, across all users."""
        if not article_ids:
            return {}

        statement = (
            select(UserEvent.article_id, func.count())
            .where(UserEvent.event_type == "click")
            .where(UserEvent.article_id.in_(article_ids))
            .group_by(UserEvent.article_id)
        )
        return {article_id: int(count) for article_id, count in self.db.execute(statement)}

    def count_events_since(self, event_type: str, since: datetime) -> int:
        statement = (
            select(func.count())
            .select_from(UserEvent)
            .where(UserEvent.event_type == event_type)
            .where(UserEvent.timestamp >= since)
        )
        return int(self.db.scalar(statement) or 0)

    def list_clicks_for_user(self, user_id: UUID, limit: int = 1000) -> list[UserEvent]:
        statement = (
            select(UserEvent)
            .where(UserEvent.user_id == user_id)
            .where(UserEvent.event_type == "click")
            .order_by(UserEvent.timestamp.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())


class RecommendationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, recommendation: Recommendation) -> Recommendation:
        self.db.add(recommendation)
        self.db.commit()
        self.db.refresh(recommendation)
        return recommendation

    def list_for_user(self, user_id: UUID, limit: int = 100) -> list[Recommendation]:
        statement = (
            select(Recommendation)
            .where(Recommendation.user_id == user_id)
            .order_by(Recommendation.rank.asc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())

    def upsert(self, recommendation: Recommendation) -> Recommendation:
        # (user_id, article_id, model_version) is unique, so refresh the score and
        # rank when the same recommendation is served again.
        statement = select(Recommendation).where(
            Recommendation.user_id == recommendation.user_id,
            Recommendation.article_id == recommendation.article_id,
            Recommendation.model_version == recommendation.model_version,
        )
        existing = self.db.scalar(statement)
        if existing is None:
            return self.create(recommendation)

        existing.score = recommendation.score
        existing.rank = recommendation.rank
        self.db.commit()
        self.db.refresh(existing)
        return existing


class ModelVersionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, model_version: ModelVersion) -> ModelVersion:
        self.db.add(model_version)
        self.db.commit()
        self.db.refresh(model_version)
        return model_version

    def get_by_version(self, version: str) -> ModelVersion | None:
        statement = select(ModelVersion).where(ModelVersion.version == version)
        return self.db.scalar(statement)
