from uuid import uuid4

from app.db.models import Article
from app.retrieval.dense import (
    ARTICLE_COLLECTION_NAME,
    article_to_embedding_text,
    article_to_payload,
    embed_texts,
    ensure_article_collection,
)


class FakeEmbedding:
    def tolist(self) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]


class FakeModel:
    def encode(self, texts: list[str], normalize_embeddings: bool) -> FakeEmbedding:
        assert texts == ["hello"]
        assert normalize_embeddings is True
        return FakeEmbedding()


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeCollectionsResponse:
    def __init__(self, names: list[str]) -> None:
        self.collections = [FakeCollection(name) for name in names]


class FakeQdrantClient:
    def __init__(self, existing_names: list[str]) -> None:
        self.existing_names = existing_names
        self.created_collection_name: str | None = None
        self.created_vectors_config = None

    def get_collections(self) -> FakeCollectionsResponse:
        return FakeCollectionsResponse(self.existing_names)

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.created_collection_name = collection_name
        self.created_vectors_config = vectors_config

def test_embed_texts_normalizes_embeddings() -> None:
    assert embed_texts(FakeModel(), ["hello"]) == [[0.1, 0.2, 0.3]]

def test_article_to_embedding_text_combines_title_text_and_category() -> None:
    article = Article(
        title="AI policy",
        url="https://example.com",
        text="Regulation news",
        source="mind",
        category="news",
    )

    assert article_to_embedding_text(article) == "AI policy\nRegulation news\nnews"

def test_article_to_payload_maps_metadata() -> None:
    article = Article(
        external_id="N123",
        title="AI policy",
        url="https://example.com",
        source="mind",
        category="news",
    )
    article.id = uuid4()

    payload = article_to_payload(article)

    assert payload["article_id"] == str(article.id)
    assert payload["external_id"] == "N123"
    assert payload["title"] == "AI policy"

def test_ensure_article_collection_creates_missing_collection() -> None:
    client = FakeQdrantClient(existing_names=[])

    ensure_article_collection(client)

    assert client.created_collection_name == ARTICLE_COLLECTION_NAME
    assert client.created_vectors_config.size == 384

def test_ensure_article_collection_skips_existing_collection() -> None:
    client = FakeQdrantClient(existing_names=[ARTICLE_COLLECTION_NAME])

    ensure_article_collection(client)

    assert client.created_collection_name is None