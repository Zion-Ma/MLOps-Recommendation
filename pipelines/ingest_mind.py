import argparse
import csv
from collections.abc import Iterator
from datetime import datetime
from itertools import islice
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Article, User, UserEvent
from app.db.repositories import ArticleRepository, UserRepository
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

DEFAULT_BATCH_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MIND dataset files.")
    parser.add_argument("--news-path", type=Path, required=True)
    parser.add_argument("--behaviors-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Commit inserts in batches of this size",
    )
    return parser.parse_args()


def parse_mind_time(value: str) -> datetime:
    return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")


def ingest_news(
    db: Session,
    news_path: Path,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, dict[str, UUID]]:
    article_repo = ArticleRepository(db)
    inserted = 0
    pending = 0
    external_id_to_article_id: dict[str, UUID] = {}

    rows = iter_tsv_rows(news_path, NEWS_COLUMNS)
    if limit is not None:
        rows = islice(rows, limit)

    for row in rows:
        article = article_from_news_row(row)
        if article is None or article.external_id is None:
            continue

        existing = article_repo.get_by_source_external_id("mind", article.external_id)
        if existing is not None:
            external_id_to_article_id[article.external_id] = existing.id
            continue

        existing_url = article_repo.get_by_url(article.url)
        if existing_url is not None:
            external_id_to_article_id[article.external_id] = existing_url.id
            continue

        db.add(article)
        db.flush()
        external_id_to_article_id[article.external_id] = article.id
        inserted += 1
        pending += 1

        if pending >= batch_size:
            db.commit()
            pending = 0

    if pending:
        db.commit()

    return inserted, external_id_to_article_id


def iter_tsv_rows(path: Path, fieldnames: list[str]) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=fieldnames, delimiter="\t")
        yield from reader


def article_from_news_row(row: dict[str, str]) -> Article | None:
    news_id = (row.get("news_id") or "").strip()
    title = (row.get("title") or "").strip()
    url = (row.get("url") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    category = (row.get("category") or "").strip()

    # Skip malformed rows (MIND occasionally embeds tabs that shift columns).
    if not news_id or not title or not url.startswith("http"):
        return None

    return Article(
        external_id=news_id,
        title=title,
        url=url,
        text=abstract or None,
        source="mind",
        category=category or None,
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


def get_or_create_user(
    db: Session,
    user_repo: UserRepository,
    external_id: str,
    cache: dict[str, User],
) -> User:
    cached = cache.get(external_id)
    if cached is not None:
        return cached

    existing = user_repo.get_by_external_id(external_id)
    if existing is not None:
        cache[external_id] = existing
        return existing

    user = User(external_id=external_id)
    db.add(user)
    db.flush()
    cache[external_id] = user
    return user


def ingest_behaviors(
    db: Session,
    behaviors_path: Path,
    article_ids: dict[str, UUID],
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    user_repo = UserRepository(db)
    inserted = 0
    pending = 0
    user_cache: dict[str, User] = {}

    rows = iter_tsv_rows(behaviors_path, BEHAVIOR_COLUMNS)
    if limit is not None:
        rows = islice(rows, limit)

    for row in rows:
        user = get_or_create_user(db, user_repo, row["user_id"], user_cache)
        event_time = parse_mind_time(row["time"])

        for position, news_id in enumerate(parse_history(row["history"]), start=1):
            article_id = article_ids.get(news_id)
            if article_id is None:
                continue

            db.add(
                UserEvent(
                    user_id=user.id,
                    article_id=article_id,
                    event_type="click",
                    position=position,
                    model_version="mind-history",
                    event_metadata={"mind_impression_id": row["impression_id"]},
                    timestamp=event_time,
                )
            )
            inserted += 1
            pending += 1

        for position, (news_id, clicked) in enumerate(
            parse_impressions(row["impressions"]),
            start=1,
        ):
            article_id = article_ids.get(news_id)
            if article_id is None:
                continue

            db.add(
                UserEvent(
                    user_id=user.id,
                    article_id=article_id,
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
            pending += 1

            if clicked == 1:
                db.add(
                    UserEvent(
                        user_id=user.id,
                        article_id=article_id,
                        event_type="click",
                        position=position,
                        model_version="mind-impression",
                        event_metadata={"mind_impression_id": row["impression_id"]},
                        timestamp=event_time,
                    )
                )
                inserted += 1
                pending += 1

        if pending >= batch_size:
            db.commit()
            pending = 0

    if pending:
        db.commit()

    return inserted


def run(
    news_path: Path,
    behaviors_path: Path,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    db = SessionLocal()
    try:
        inserted_articles, article_ids = ingest_news(
            db, news_path, limit=limit, batch_size=batch_size
        )
        inserted_events = ingest_behaviors(
            db,
            behaviors_path,
            article_ids=article_ids,
            limit=limit,
            batch_size=batch_size,
        )
        return inserted_articles, inserted_events
    finally:
        db.close()


def main() -> None:
    args = parse_args()
    inserted_articles, inserted_events = run(
        news_path=args.news_path,
        behaviors_path=args.behaviors_path,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(f"Inserted {inserted_articles} MIND articles.")
    print(f"Inserted {inserted_events} MIND user events.")


if __name__ == "__main__":
    main()
