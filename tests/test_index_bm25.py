from types import SimpleNamespace

from pipelines import index_bm25


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


def test_run_indexes_recent_articles_and_closes_db(monkeypatch) -> None:
    session = FakeSession()
    indexed_articles: list[object] = []

    def fake_index_articles(client: object, articles: list[object]) -> int:
        indexed_articles.extend(articles)
        return len(articles)

    monkeypatch.setattr(index_bm25, "SessionLocal", lambda: session)
    monkeypatch.setattr(index_bm25, "ArticleRepository", FakeArticleRepository)
    monkeypatch.setattr(index_bm25, "create_opensearch_client", lambda: object())
    monkeypatch.setattr(index_bm25, "index_articles", fake_index_articles)

    indexed = index_bm25.run(limit=3)

    assert indexed == 3
    assert len(indexed_articles) == 3
    assert session.closed is True
