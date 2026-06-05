from datetime import datetime

from pipelines.ingest_hn import ingest_stories, normalize_story


class FakeArticleRepository:
    created_urls: set[str] = set()

    def __init__(self, db: object) -> None:
        self.db = db

    def get_by_url(self, url: str) -> object | None:
        if url in self.created_urls:
            return object()
        return None

    def create(self, article: object) -> object:
        self.created_urls.add(article.url)
        return article


def test_normalize_story_maps_hn_payload_to_article() -> None:
    article = normalize_story(
        {
            "objectID": "123",
            "title": "A useful article",
            "url": "https://example.com/article",
            "story_text": "summary",
            "created_at": "2026-05-14T12:30:00Z",
        }
    )

    assert article is not None
    assert article.title == "A useful article"
    assert article.url == "https://example.com/article"
    assert article.text == "summary"
    assert article.source == "hacker_news"
    assert article.category == "technology"
    assert article.published_at == datetime.fromisoformat("2026-05-14T12:30:00+00:00")


def test_normalize_story_falls_back_to_hn_item_url() -> None:
    article = normalize_story(
        {
            "objectID": "123",
            "title": "Ask HN style story",
            "created_at": "2026-05-14T12:30:00Z",
        }
    )

    assert article is not None
    assert article.url == "https://news.ycombinator.com/item?id=123"


def test_normalize_story_skips_records_without_title() -> None:
    article = normalize_story({"objectID": "123", "url": "https://example.com/article"})

    assert article is None


def test_ingest_stories_deduplicates_by_url(monkeypatch) -> None:
    FakeArticleRepository.created_urls = set()
    monkeypatch.setattr("pipelines.ingest_hn.ArticleRepository", FakeArticleRepository)

    inserted = ingest_stories(
        db=object(),
        stories=[
            {
                "objectID": "1",
                "title": "First",
                "url": "https://example.com/article",
            },
            {
                "objectID": "2",
                "title": "Duplicate",
                "url": "https://example.com/article",
            },
        ],
    )

    assert inserted == 1
