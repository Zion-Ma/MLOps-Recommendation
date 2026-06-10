import argparse

from app.db.repositories import ArticleRepository
from app.db.session import SessionLocal
from app.retrieval.bm25 import create_opensearch_client, index_articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index Postgres articles into OpenSearch.")
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def run(limit: int) -> int:
    db = SessionLocal()
    try:
        articles = ArticleRepository(db).list_recent(limit=limit)
        client = create_opensearch_client()
        return index_articles(client=client, articles=articles)
    finally:
        db.close()


def main() -> None:
    args = parse_args()
    indexed = run(limit=args.limit)
    print(f"Indexed {indexed} articles into OpenSearch.")


if __name__ == "__main__":
    main()
