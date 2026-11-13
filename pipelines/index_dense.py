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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Articles embedded and upserted per batch (keeps memory and request sizes bounded)",
    )
    return parser.parse_args()


def run(limit: int, batch_size: int = 2000) -> int:
    db = SessionLocal()
    try:
        articles = ArticleRepository(db).list_recent(limit=limit)
        client = create_qdrant_client()
        model = create_embedding_model()

        indexed = 0
        for start in range(0, len(articles), batch_size):
            batch = articles[start : start + batch_size]
            indexed += index_article_embeddings(client=client, model=model, articles=batch)
            print(f"Indexed {indexed}/{len(articles)} article embeddings", flush=True)
        return indexed
    finally:
        db.close()


def main() -> None:
    args = parse_args()
    indexed = run(limit=args.limit, batch_size=args.batch_size)
    print(f"Indexed {indexed} article embeddings into Qdrant.")


if __name__ == "__main__":
    main()
