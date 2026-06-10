import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.db.models import Article

ARTICLE_INDEX_NAME = "articles"

# OpenSearch uses BM25 for text fields by default, but keeping the mapping explicit
# makes the retrieval behavior easier to reason about and reproduce.
ARTICLE_INDEX_MAPPING: dict[str, Any] = {
    "settings": {
        "index": {
            "similarity": {
                "default": {
                    "type": "BM25",
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "article_id": {"type": "keyword"},
            "external_id": {"type": "keyword"},
            "title": {"type": "text"},
            "text": {"type": "text"},
            "url": {"type": "keyword"},
            "source": {"type": "keyword"},
            "category": {"type": "keyword"},
            "published_at": {"type": "date"},
        }
    },
}


@dataclass(frozen=True)
class BM25Result:
    article_id: UUID
    score: float
    title: str
    url: str
    source: str
    category: str | None


def create_opensearch_client() -> Any:
    from opensearchpy import OpenSearch  # type: ignore[import-not-found]

    # Docker Compose exposes OpenSearch as "opensearch"; local CLI usage defaults to localhost.
    return OpenSearch(
        hosts=[os.getenv("OPENSEARCH_URL", "http://localhost:9200")],
        verify_certs=False,
    )


def ensure_article_index(client: Any, index_name: str = ARTICLE_INDEX_NAME) -> None:
    # The index is derived from Postgres articles, so creation is idempotent.
    if client.indices.exists(index=index_name):
        return
    client.indices.create(index=index_name, body=ARTICLE_INDEX_MAPPING)


def article_to_document(article: Article) -> dict[str, Any]:
    # Store only retrieval-facing fields in OpenSearch; Postgres remains the source of truth.
    return {
        "article_id": str(article.id),
        "external_id": article.external_id,
        "title": article.title,
        "text": article.text,
        "url": article.url,
        "source": article.source,
        "category": article.category,
        "published_at": article.published_at.isoformat() if article.published_at else None,
    }


def index_articles(
    client: Any,
    articles: list[Article],
    index_name: str = ARTICLE_INDEX_NAME,
) -> int:
    ensure_article_index(client, index_name=index_name)

    # Use the Postgres article UUID as the OpenSearch document ID to make re-indexing stable.
    for article in articles:
        client.index(
            index=index_name,
            id=str(article.id),
            body=article_to_document(article),
            refresh=False,
        )

    if articles:
        client.indices.refresh(index=index_name)

    return len(articles)


def search_bm25(
    client: Any,
    query: str,
    top_k: int = 10,
    index_name: str = ARTICLE_INDEX_NAME,
) -> list[BM25Result]:
    # Title matches are weighted higher because they are usually more precise than body text.
    body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "text", "category^2"],
            }
        },
    }

    response = client.search(index=index_name, body=body)
    hits = response.get("hits", {}).get("hits", [])

    return [_result_from_hit(hit) for hit in hits]


def _result_from_hit(hit: dict[str, Any]) -> BM25Result:
    source = hit["_source"]
    return BM25Result(
        article_id=UUID(source["article_id"]),
        score=float(hit["_score"]),
        title=source["title"],
        url=source["url"],
        source=source["source"],
        category=source.get("category"),
    )
