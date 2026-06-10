import argparse

from app.retrieval.bm25 import create_opensearch_client, search_bm25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search articles with OpenSearch BM25.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def run(query: str, top_k: int) -> None:
    client = create_opensearch_client()
    results = search_bm25(client=client, query=query, top_k=top_k)

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.title}")
        print(f"   score={result.score:.4f} source={result.source} category={result.category}")
        print(f"   url={result.url}")


def main() -> None:
    args = parse_args()
    run(query=args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()