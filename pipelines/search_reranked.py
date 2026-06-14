import argparse

from app.db.repositories import ArticleRepository
from app.db.session import SessionLocal
from app.retrieval.bm25 import create_opensearch_client, search_bm25
from app.retrieval.dense import article_to_embedding_text, create_embedding_model
from app.retrieval.rerank import rerank_bm25_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search articles with BM25 plus dense reranking.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def run(query: str, candidate_pool_size: int, top_k: int) -> None:
    opensearch_client = create_opensearch_client()
    model = create_embedding_model()
    bm25_results = search_bm25(
        client=opensearch_client,
        query=query,
        top_k=candidate_pool_size,
    )
    if not bm25_results:
        print("No BM25 candidates found.")
        return

    article_ids = [result.article_id for result in bm25_results]
    db = SessionLocal()
    try:
        articles = ArticleRepository(db).get_by_ids(article_ids)
    finally:
        db.close()

    article_texts = {article.id: article_to_embedding_text(article) for article in articles}
    missing_count = len(article_ids) - len(article_texts)
    if missing_count:
        print(f"Skipped {missing_count} BM25 candidates missing from Postgres.")

    reranked_results = rerank_bm25_results(
        query=query,
        bm25_results=bm25_results,
        article_texts=article_texts,
        model=model,
        top_k=top_k,
    )
    if not reranked_results:
        print("No reranked results found.")
        return

    for index, result in enumerate(reranked_results, start=1):
        print(f"{index}. {result.title}")
        print(
            f"   rerank={result.rerank_score:.4f} "
            f"dense={result.dense_score:.4f} "
            f"bm25={result.bm25_score:.4f} "
            f"source={result.source} "
            f"category={result.category}"
        )
        print(f"   url={result.url}")


def main() -> None:
    args = parse_args()
    run(
        query=args.query,
        candidate_pool_size=args.candidate_pool_size,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
