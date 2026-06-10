import os
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.db.models import Article

ARTICLE_COLLECTION_NAME = "article_embeddings"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_SIZE = 384


@dataclass(frozen=True)
class DenseResult:
    article_id: UUID
    score: float
    title: str
    url: str
    source: str
    category: str | None


# Create Qdrant client
def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )


# Create embedding model
def create_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    return cast(SentenceTransformer, SentenceTransformer(model_name))


# Embed texts
def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# Ensure Qdrant collection
def ensure_article_collection(
    client: QdrantClient,
    collection_name: str = ARTICLE_COLLECTION_NAME,
    vector_size: int = DEFAULT_VECTOR_SIZE,
) -> None:
    existing = [collection.name for collection in client.get_collections().collections]
    if collection_name in existing:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


# Convert Article to embedding text
def article_to_embedding_text(article: Article) -> str:
    parts = [
        article.title,
        article.text or "",
        article.category or "",
    ]
    return "\n".join(part for part in parts if part)


# Convert Article to payload
def article_to_payload(article: Article) -> dict[str, Any]:
    return {
        "article_id": str(article.id),
        "external_id": article.external_id,
        "title": article.title,
        "url": article.url,
        "source": article.source,
        "category": article.category,
    }


# Index article embeddings
def index_article_embeddings(
    client: QdrantClient,
    model: SentenceTransformer,
    articles: list[Article],
) -> int:
    ensure_article_collection(client)

    texts = [article_to_embedding_text(article) for article in articles]
    vectors = embed_texts(model, texts)

    points = [
        PointStruct(
            id=str(article.id),
            vector=vector,
            payload=article_to_payload(article),
        )
        for article, vector in zip(articles, vectors, strict=True)
    ]

    client.upsert(collection_name=ARTICLE_COLLECTION_NAME, points=points)

    return len(points)


# Search for similar articles using dense embeddings
def search_dense(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    top_k: int = 10,
    collection_name: str = ARTICLE_COLLECTION_NAME,
) -> list[DenseResult]:
    query_vector = embed_texts(model, [query])[0]
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [_result_from_point(point) for point in response.points]


# Convert a Qdrant point to a DenseResult
def _result_from_point(point: Any) -> DenseResult:
    payload = point.payload or {}

    return DenseResult(
        article_id=UUID(str(payload["article_id"])),
        score=float(point.score),
        title=str(payload["title"]),
        url=str(payload["url"]),
        source=str(payload["source"]),
        category=str(payload["category"]) if payload.get("category") is not None else None,
    )
