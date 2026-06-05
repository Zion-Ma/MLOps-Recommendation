import argparse
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.db.models import Article
from app.db.repositories import ArticleRepository
from app.db.session import SessionLocal

HN_SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_ITEM_URL = "https://news.ycombinator.com/item"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest recent Hacker News stories.")
    parser.add_argument("--days-back", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def fetch_recent_stories(days_back: int, limit: int) -> list[dict[str, Any]]:
    since = datetime.now(tz=UTC) - timedelta(days=days_back)
    params: dict[str, str | int] = {
        "tags": "story",
        "numericFilters": f"created_at_i>{int(since.timestamp())}",
        "hitsPerPage": limit,
    }

    response = httpx.get(HN_SEARCH_BY_DATE_URL, params=params, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("hits", [])

    if not isinstance(hits, list):
        msg = "Hacker News API response did not include a list of hits"
        raise ValueError(msg)

    return hits


def normalize_story(story: dict[str, Any]) -> Article | None:
    title = _get_string(story, "title") or _get_string(story, "story_title")
    object_id = _get_string(story, "objectID")

    if not title or not object_id:
        return None

    url = _get_string(story, "url") or f"{HN_ITEM_URL}?id={object_id}"
    published_at = _parse_datetime(_get_string(story, "created_at"))

    return Article(
        title=title,
        url=url,
        text=_get_string(story, "story_text"),
        source="hacker_news",
        published_at=published_at,
        category="technology",
    )


def ingest_stories(db: Session, stories: list[dict[str, Any]]) -> int:
    repo = ArticleRepository(db)
    inserted = 0

    for story in stories:
        article = normalize_story(story)
        if article is None or repo.get_by_url(article.url) is not None:
            continue

        repo.create(article)
        inserted += 1

    return inserted


def run(days_back: int, limit: int) -> int:
    stories = fetch_recent_stories(days_back=days_back, limit=limit)
    db = SessionLocal()
    try:
        return ingest_stories(db=db, stories=stories)
    finally:
        db.close()


def _get_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    args = parse_args()
    inserted = run(days_back=args.days_back, limit=args.limit)
    print(f"Inserted {inserted} Hacker News articles.")


if __name__ == "__main__":
    main()
