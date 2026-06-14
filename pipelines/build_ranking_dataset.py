import argparse
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Article, UserEvent
from app.db.repositories import ArticleRepository, UserEventRepository
from app.db.session import SessionLocal
from app.features.ranking import CandidateScores, RankingFeatureRow, build_ranking_feature_row


def event_label(event: UserEvent) -> int | None:
    if event.event_metadata is None:
        return None
    clicked = event.event_metadata.get("clicked")
    if clicked not in (0, 1):
        return None
    return int(clicked)

# convert each row to a plain dictionary before writing Parquet
def row_to_dict(row: RankingFeatureRow) -> dict[str, object]:
    record = asdict(row)
    record["user_id"] = str(record["user_id"])
    record["article_id"] = str(record["article_id"])
    return record

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ranking feature dataset.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()

def build_rows(db: Session, limit: int) -> list[RankingFeatureRow]:
    event_repo = UserEventRepository(db)
    article_repo = ArticleRepository(db)

    impression_events = event_repo.list_labeled_impressions(limit=limit)
    if not impression_events:
        return []

    article_ids = [event.article_id for event in impression_events]
    articles = article_repo.get_by_ids(article_ids)
    articles_by_id = {article.id: article for article in articles}

    rows = []
    for event in impression_events:
        label = event_label(event)
        article = articles_by_id.get(event.article_id)

        if label is None or article is None:
            continue

        user_clicks = event_repo.list_clicks_for_user(event.user_id)

        clicked_article_ids = [click.article_id for click in user_clicks]
        clicked_categories = _clicked_categories(
            clicks=user_clicks,
            articles_by_id=articles_by_id,
        )

        rows.append(
            build_ranking_feature_row(
                group_id=str(event.event_metadata["mind_impression_id"]),
                user_id=event.user_id,
                candidate=CandidateScores(
                    article_id=event.article_id,
                    bm25_score=0.0,
                    dense_score=0.0,
                    rerank_score=0.0,
                ),
                published_at=article.published_at,
                candidate_category=article.category,
                clicked_article_ids=clicked_article_ids,
                clicked_categories=clicked_categories,
                label=label,
            )
        )

    return rows

def _clicked_categories(
    clicks: list[UserEvent],
    articles_by_id: dict[UUID, Article],
) -> list[str]:
    categories = []

    for click in clicks:
        article = articles_by_id.get(click.article_id)
        if article is not None and article.category is not None:
            categories.append(article.category)

    return categories

def write_parquet(rows: list[RankingFeatureRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [row_to_dict(row) for row in rows]
    dataframe = pd.DataFrame.from_records(records)
    dataframe.to_parquet(output_path, index=False)

def run(output_path: Path, limit: int) -> int:
    db = SessionLocal()
    try:
        rows = build_rows(db, limit)
        write_parquet(rows, output_path)
        return len(rows)
    finally:
        db.close()

def main() -> None:
    args = parse_args()
    row_count = run(Path(args.output_path), args.limit)
    print(f"Wrote {row_count} ranking feature rows to {args.output_path}.")

if __name__ == "__main__":
    main()