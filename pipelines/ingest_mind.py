import argparse
import csv
from collections.abc import Iterator
from datetime import datetime
from itertools import islice
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Article, User, UserEvent
from app.db.repositories import ArticleRepository, UserEventRepository, UserRepository
from app.db.session import SessionLocal

"""
1. Parse news.tsv
2. Parse behaviors.tsv
3. Normalize rows into Article/User/UserEvent objects
4. Insert them into Postgres
"""

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MIND dataset files.")
    parser.add_argument("--news-path", type=Path, required=True)
    parser.add_argument("--behaviors-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()

def parse_mind_time(value: str) -> datetime:
    return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")


def ingest_news(db: Session, news_path: Path, limit: int | None = None) -> int:
    article_repo = ArticleRepository(db)
    inserted = 0

    rows = iter_tsv_rows(news_path, NEWS_COLUMNS)
    if limit is not None:
        rows = islice(rows, limit)

    for row in rows:
        article = article_from_news_row(row)
        if article is None or article.external_id is None:
            continue

        existing = article_repo.get_by_source_external_id("mind", article.external_id)
        if existing is not None:
            continue

        article_repo.create(article)
        inserted += 1

    return inserted

# Row Parser Functions
def iter_tsv_rows(path: Path, fieldnames: list[str]) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=fieldnames, delimiter="\t")
        yield from reader

# Normalize News Row
def article_from_news_row(row: dict[str, str]) -> Article | None:
    news_id = row["news_id"].strip()
    title = row["title"].strip()
    url = row["url"].strip()

    if not news_id or not title or not url:
        return None

    return Article(
        external_id=news_id,
        title=title,
        url=url,
        text=row["abstract"].strip() or None,
        source="mind",
        category=row["category"].strip() or None,
    )

def parse_history(history: str) -> list[str]:
    if not history.strip():
        return []
    return history.split()


def parse_impressions(impressions: str) -> list[tuple[str, int]]:
    parsed = []
    for item in impressions.split():
        news_id, label = item.rsplit("-", 1)
        parsed.append((news_id, int(label)))
    return parsed

def get_or_create_user(user_repo: UserRepository, external_id: str) -> User:
    existing = user_repo.get_by_external_id(external_id)
    if existing is not None:
        return existing
    return user_repo.create(User(external_id=external_id))


def ingest_behaviors(db: Session, behaviors_path: Path, limit: int | None = None) -> int:
    article_repo = ArticleRepository(db)
    user_repo = UserRepository(db)
    event_repo = UserEventRepository(db)
    inserted = 0

    rows = iter_tsv_rows(behaviors_path, BEHAVIOR_COLUMNS)
    if limit is not None:
        rows = islice(rows, limit)

    for row in rows:
        user = get_or_create_user(user_repo, row["user_id"])
        event_time = parse_mind_time(row["time"])

        for position, news_id in enumerate(parse_history(row["history"]), start=1):
            article = article_repo.get_by_source_external_id("mind", news_id)
            if article is None:
                continue

            event_repo.create(
                UserEvent(
                    user_id=user.id,
                    article_id=article.id,
                    event_type="click",
                    position=position,
                    model_version="mind-history",
                    event_metadata={"mind_impression_id": row["impression_id"]},
                    timestamp=event_time,
                )
            )
            inserted += 1

        for position, (news_id, clicked) in enumerate(
            parse_impressions(row["impressions"]),
            start=1,
        ):
            article = article_repo.get_by_source_external_id("mind", news_id)
            if article is None:
                continue

            event_repo.create(
                UserEvent(
                    user_id=user.id,
                    article_id=article.id,
                    event_type="impression",
                    position=position,
                    model_version="mind-impression",
                    event_metadata={
                        "mind_impression_id": row["impression_id"],
                        "clicked": clicked,
                    },
                    timestamp=event_time,
                )
            )
            inserted += 1

            if clicked == 1:
                event_repo.create(
                    UserEvent(
                        user_id=user.id,
                        article_id=article.id,
                        event_type="click",
                        position=position,
                        model_version="mind-impression",
                        event_metadata={"mind_impression_id": row["impression_id"]},
                        timestamp=event_time,
                    )
                )
                inserted += 1

    return inserted

def run(news_path: Path, behaviors_path: Path, limit: int | None = None) -> tuple[int, int]:
    db = SessionLocal()
    try:
        inserted_articles = ingest_news(db, news_path, limit)
        inserted_events = ingest_behaviors(db, behaviors_path, limit)
        return inserted_articles, inserted_events
    finally:
        db.close()


def main() -> None:
    args = parse_args()
    inserted_articles, inserted_events = run(
        news_path=args.news_path,
        behaviors_path=args.behaviors_path,
        limit=args.limit,
    )
    print(f"Inserted {inserted_articles} MIND articles.")
    print(f"Inserted {inserted_events} MIND user events.")


if __name__ == "__main__":
    main()