from uuid import uuid4

from app.db.models import Article
from app.retrieval.bm25 import (
    ARTICLE_INDEX_MAPPING,
    article_to_document,
    ensure_article_index,
    index_articles,
    search_bm25,
)


class FakeIndices:
    def __init__(self, exists: bool = False) -> None:
        self.exists_value = exists
        self.created_body: dict | None = None
        self.refreshed_index: str | None = None

    def exists(self, index: str) -> bool:
        return self.exists_value

    def create(self, index: str, body: dict) -> None:
        self.created_body = body

    def refresh(self, index: str) -> None:
        self.refreshed_index = index


class FakeOpenSearchClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.indexed_documents: list[dict] = []
        self.search_body: dict | None = None

    def index(self, index: str, id: str, body: dict, refresh: bool) -> None:
        self.indexed_documents.append(
            {
                "index": index,
                "id": id,
                "body": body,
                "refresh": refresh,
            }
        )

    def search(self, index: str, body: dict) -> dict:
        self.search_body = body
        article_id = str(uuid4())
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 12.5,
                        "_source": {
                            "article_id": article_id,
                            "title": "Vector search for news",
                            "url": "https://example.com/vector-search",
                            "source": "mind",
                            "category": "technology",
                        },
                    }
                ]
            }
        }


def test_ensure_article_index_creates_mapping_when_missing() -> None:
    client = FakeOpenSearchClient()

    ensure_article_index(client)

    assert client.indices.created_body == ARTICLE_INDEX_MAPPING


def test_article_to_document_maps_article_fields() -> None:
    article = Article(
        external_id="N123",
        title="Ranking systems",
        url="https://example.com/ranking",
        text="A ranking article",
        source="mind",
        category="technology",
    )
    article.id = uuid4()

    document = article_to_document(article)

    assert document["article_id"] == str(article.id)
    assert document["external_id"] == "N123"
    assert document["title"] == "Ranking systems"
    assert document["text"] == "A ranking article"
    assert document["source"] == "mind"


def test_index_articles_indexes_and_refreshes() -> None:
    client = FakeOpenSearchClient()
    article = Article(title="BM25", url="https://example.com/bm25", source="mind")
    article.id = uuid4()

    indexed_count = index_articles(client, [article])

    assert indexed_count == 1
    assert client.indexed_documents[0]["id"] == str(article.id)
    assert client.indices.refreshed_index == "articles"


def test_search_bm25_builds_multi_match_query_and_parses_results() -> None:
    client = FakeOpenSearchClient()

    results = search_bm25(client, query="vector search", top_k=5)

    assert client.search_body == {
        "size": 5,
        "query": {
            "multi_match": {
                "query": "vector search",
                "fields": ["title^3", "text", "category^2"],
            }
        },
    }
    assert len(results) == 1
    assert results[0].title == "Vector search for news"
    assert results[0].score == 12.5
