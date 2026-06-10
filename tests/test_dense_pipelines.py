from types import SimpleNamespace

from pipelines import index_dense, search_dense


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeArticleRepository:
    def __init__(self, db: FakeSession) -> None:
        self.db = db

    def list_recent(self, limit: int) -> list[object]:
        return [SimpleNamespace(id="article-1") for _ in range(limit)]


def test_index_dense_run_indexes_recent_articles_and_closes_db(monkeypatch) -> None:
    session = FakeSession()
    indexed_articles: list[object] = []

    def fake_index_article_embeddings(
        client: object,
        model: object,
        articles: list[object],
    ) -> int:
        indexed_articles.extend(articles)
        return len(articles)

    monkeypatch.setattr(index_dense, "SessionLocal", lambda: session)
    monkeypatch.setattr(index_dense, "ArticleRepository", FakeArticleRepository)
    monkeypatch.setattr(index_dense, "create_qdrant_client", lambda: object())
    monkeypatch.setattr(index_dense, "create_embedding_model", lambda: object())
    monkeypatch.setattr(index_dense, "index_article_embeddings", fake_index_article_embeddings)

    indexed = index_dense.run(limit=3)

    assert indexed == 3
    assert len(indexed_articles) == 3
    assert session.closed is True


def test_search_dense_run_prints_results(monkeypatch, capsys) -> None:
    result = SimpleNamespace(
        title="Semantic article",
        score=0.91,
        source="mind",
        category="technology",
        url="https://example.com/semantic",
    )

    monkeypatch.setattr(search_dense, "create_qdrant_client", lambda: object())
    monkeypatch.setattr(search_dense, "create_embedding_model", lambda: object())
    monkeypatch.setattr(search_dense, "search_dense", lambda **_: [result])

    search_dense.run(query="semantic search", top_k=1)

    output = capsys.readouterr().out
    assert "Semantic article" in output
    assert "score=0.9100" in output
    assert "https://example.com/semantic" in output
