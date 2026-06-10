import argparse

from app.retrieval.dense import create_embedding_model, create_qdrant_client, search_dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search articles with Qdrant dense retrieval.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def run(query: str, top_k: int) -> None:
    client = create_qdrant_client()
    model = create_embedding_model()
    results = search_dense(client=client, model=model, query=query, top_k=top_k)

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.title}")
        print(f"   score={result.score:.4f} source={result.source} category={result.category}")
        print(f"   url={result.url}")


def main() -> None:
    args = parse_args()
    run(query=args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
