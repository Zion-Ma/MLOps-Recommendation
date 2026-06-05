from uuid import UUID

from sqlalchemy import select
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
