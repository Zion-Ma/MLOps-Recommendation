from datetime import datetime

from pipelines.ingest_mind import (
    article_from_news_row,
    parse_history,
    parse_impressions,
    parse_mind_time,
)


def test_article_from_news_row() -> None:
    article = article_from_news_row(
        {
            "news_id": "N1",
            "category": "sports",
            "subcategory": "football",
            "title": "Match report",
            "abstract": "Team wins",
            "url": "https://example.com/n1",
            "title_entities": "",
            "abstract_entities": "",
        }
    )

    assert article is not None
    assert article.external_id == "N1"
    assert article.source == "mind"
    assert article.category == "sports"
    assert article.text == "Team wins"


def test_article_from_news_row_skips_incomplete() -> None:
    assert article_from_news_row(
        {
            "news_id": "",
            "category": "sports",
            "subcategory": "",
            "title": "Missing id",
            "abstract": "",
            "url": "https://example.com/n1",
            "title_entities": "",
            "abstract_entities": "",
        }
    ) is None


def test_article_from_news_row_skips_shifted_url() -> None:
    assert article_from_news_row(
        {
            "news_id": "N1",
            "category": "sports",
            "subcategory": "",
            "title": "Broken row",
            "abstract": "",
            "url": "[]",
            "title_entities": "",
            "abstract_entities": "",
        }
    ) is None


def test_parse_history_and_impressions() -> None:
    assert parse_history("N1 N2") == ["N1", "N2"]
    assert parse_history("") == []
    assert parse_impressions("N1-1 N2-0") == [("N1", 1), ("N2", 0)]


def test_parse_mind_time() -> None:
    assert parse_mind_time("11/15/2019 8:33:25 AM") == datetime(2019, 11, 15, 8, 33, 25)
