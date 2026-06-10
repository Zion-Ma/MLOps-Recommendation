import argparse

from app.db.repositories import ArticleRepository
from app.db.session import SessionLocal
from app.retrieval.dense import (
    create_embedding_model,
    create_qdrant_client,
    index_article_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index Postgres articles into Qdrant.")
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def run(limit: int) -> int:
    db = SessionLocal()
    try:
        articles = ArticleRepository(db).list_recent(limit=limit)
        client = create_qdrant_client()
        model = create_embedding_model()
        return index_article_embeddings(client=client, model=model, articles=articles)
    finally:
        db.close()


def main() -> None:
    args = parse_args()
    indexed = run(limit=args.limit)
    print(f"Indexed {indexed} article embeddings into Qdrant.")


if __name__ == "__main__":
    main()
