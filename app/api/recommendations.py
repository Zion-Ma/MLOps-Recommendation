import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import lightgbm as lgb
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.api.cache import cache_get, cache_set, cache_ttl_seconds, get_cache_client
from app.db.models import Article, Recommendation, User
from app.db.repositories import (
    ArticleRepository,
    RecommendationRepository,
    UserEventRepository,
    UserRepository,
)
from app.db.session import get_db
from app.features.ranking import CandidateScores, build_ranking_feature_row
from app.monitoring.metrics import RANKING_LATENCY, RECOMMENDATIONS_SERVED, RETRIEVAL_LATENCY
from app.ranking.lightgbm_ranker import FEATURE_COLUMNS, load_model
from app.retrieval.bm25 import create_opensearch_client, search_bm25
from app.retrieval.dense import (
    article_to_embedding_text,
    create_embedding_model,
    create_qdrant_client,
    fetch_article_vectors,
)
from app.retrieval.rerank import rerank_bm25_results_precomputed

router = APIRouter(tags=["recommendations"])

DEFAULT_MODEL_VERSION = "ranker-local"
QUERY_CLICK_COUNT = 5


class RecommendedArticle(BaseModel):
    article_id: UUID
    title: str
    url: str
    source: str
    category: str | None
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: UUID
    model_version: str
    strategy: str
    items: list[RecommendedArticle]


@lru_cache(maxsize=1)
def get_search_client() -> Any:
    return create_opensearch_client()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return create_embedding_model()


@lru_cache(maxsize=1)
def get_qdrant_client() -> Any:
    return create_qdrant_client()


@lru_cache(maxsize=1)
def get_ranker() -> lgb.Booster:
    model_path = Path(os.getenv("RANKER_MODEL_PATH", "models/ranker.txt"))
    return load_model(model_path)


def get_model_version() -> str:
    return os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION)


def resolve_user(db: Session, user_id: str) -> User | None:
    try:
        return db.get(User, UUID(user_id))
    except ValueError:
        return UserRepository(db).get_by_external_id(user_id)


def build_user_query(clicked_articles: list[Article]) -> str:
    parts: list[str] = []
    for article in clicked_articles[:QUERY_CLICK_COUNT]:
        parts.append(article.title)
        if article.category:
            parts.append(article.category)
    return " ".join(parts)


def _recency_fallback(
    db: Session,
    user: User,
    model_version: str,
    top_k: int,
) -> RecommendationResponse:
    articles = ArticleRepository(db).list_recent(limit=top_k)
    items = [
        RecommendedArticle(
            article_id=article.id,
            title=article.title,
            url=article.url,
            source=article.source,
            category=article.category,
            score=0.0,
            rank=rank,
        )
        for rank, article in enumerate(articles, start=1)
    ]
    return RecommendationResponse(
        user_id=user.id,
        model_version=model_version,
        strategy="recency_fallback",
        items=items,
    )


def recommend_for_user(
    db: Session,
    user: User,
    *,
    search_client: Any,
    embedding_model: SentenceTransformer,
    ranker: lgb.Booster,
    model_version: str,
    top_k: int,
    candidate_pool_size: int,
    vector_client: Any = None,
) -> RecommendationResponse:
    """Full serving pipeline: history -> retrieval -> features -> ranking."""
    event_repo = UserEventRepository(db)
    article_repo = ArticleRepository(db)

    clicks = event_repo.list_clicks_for_user(user.id, limit=100)
    clicked_article_ids = [click.article_id for click in clicks]
    clicked_articles = article_repo.get_by_ids(clicked_article_ids)
    clicked_by_id = {article.id: article for article in clicked_articles}
    # Preserve click recency order for query construction.
    ordered_clicked = [
        clicked_by_id[article_id]
        for article_id in clicked_article_ids
        if article_id in clicked_by_id
    ]

    query = build_user_query(ordered_clicked)
    if not query:
        return _recency_fallback(db, user, model_version, top_k)

    with RETRIEVAL_LATENCY.labels(stage="bm25").time():
        bm25_results = search_bm25(search_client, query, top_k=candidate_pool_size)

    # Do not recommend articles the user already clicked.
    seen_ids = set(clicked_article_ids)
    bm25_results = [result for result in bm25_results if result.article_id not in seen_ids]
    if not bm25_results:
        return _recency_fallback(db, user, model_version, top_k)

    candidate_articles = article_repo.get_by_ids([result.article_id for result in bm25_results])
    articles_by_id = {article.id: article for article in candidate_articles}
    article_texts = {
        article.id: article_to_embedding_text(article) for article in candidate_articles
    }

    # Precomputed candidate embeddings avoid re-encoding ~candidate_pool_size
    # texts per request; missing vectors fall back to on-the-fly encoding.
    candidate_vectors: dict[UUID, list[float]] = {}
    if vector_client is not None:
        try:
            candidate_vectors = fetch_article_vectors(
                vector_client,
                [result.article_id for result in bm25_results],
            )
        except Exception:
            candidate_vectors = {}

    with RETRIEVAL_LATENCY.labels(stage="rerank").time():
        reranked = rerank_bm25_results_precomputed(
            query=query,
            bm25_results=bm25_results,
            candidate_vectors=candidate_vectors,
            article_texts=article_texts,
            model=embedding_model,
            top_k=len(bm25_results),
        )

    clicked_categories = [
        article.category for article in ordered_clicked if article.category is not None
    ]
    # Global popularity, matching how the training dataset computes it.
    popularity_by_article = event_repo.count_clicks_by_article(
        [result.article_id for result in reranked]
    )

    feature_rows = []
    candidates = []
    for result in reranked:
        article = articles_by_id.get(result.article_id)
        if article is None:
            continue
        candidates.append(result)
        feature_rows.append(
            build_ranking_feature_row(
                group_id=str(user.id),
                user_id=user.id,
                candidate=CandidateScores(
                    article_id=result.article_id,
                    bm25_score=result.bm25_score,
                    dense_score=result.dense_score,
                    rerank_score=result.rerank_score,
                ),
                published_at=article.published_at,
                candidate_category=article.category,
                clicked_article_ids=clicked_article_ids,
                clicked_categories=clicked_categories,
                article_popularity=float(popularity_by_article.get(result.article_id, 0)),
            )
        )

    if not candidates:
        return _recency_fallback(db, user, model_version, top_k)

    features = pd.DataFrame(
        [
            {
                "bm25_score": row.bm25_score,
                "dense_score": row.dense_score,
                "rerank_score": row.rerank_score,
                "article_age_hours": row.article_age_hours,
                "article_popularity": row.article_popularity,
                "user_category_affinity": row.user_category_affinity,
            }
            for row in feature_rows
        ],
        columns=FEATURE_COLUMNS,
    )
    features["article_age_hours"] = features["article_age_hours"].fillna(0.0)

    with RANKING_LATENCY.time():
        scores = [float(score) for score in ranker.predict(features)]

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]

    recommendation_repo = RecommendationRepository(db)
    items = []
    for rank, (candidate, score) in enumerate(ranked, start=1):
        recommendation_repo.upsert(
            Recommendation(
                user_id=user.id,
                article_id=candidate.article_id,
                model_version=model_version,
                score=score,
                rank=rank,
            )
        )
        items.append(
            RecommendedArticle(
                article_id=candidate.article_id,
                title=candidate.title,
                url=candidate.url,
                source=candidate.source,
                category=candidate.category,
                score=score,
                rank=rank,
            )
        )

    return RecommendationResponse(
        user_id=user.id,
        model_version=model_version,
        strategy="ranked",
        items=items,
    )


@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    db: Annotated[Session, Depends(get_db)],
    search_client: Annotated[Any, Depends(get_search_client)],
    embedding_model: Annotated[SentenceTransformer, Depends(get_embedding_model)],
    ranker: Annotated[lgb.Booster, Depends(get_ranker)],
    model_version: Annotated[str, Depends(get_model_version)],
    cache_client: Annotated[Any, Depends(get_cache_client)],
    vector_client: Annotated[Any, Depends(get_qdrant_client)],
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
    candidate_pool_size: Annotated[int, Query(ge=10, le=1000)] = 200,
) -> RecommendationResponse:
    user = resolve_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {user_id!r} not found")

    cache_key = f"recommendations:{user.id}:{model_version}:{top_k}:{candidate_pool_size}"
    cached = cache_get(cache_client, cache_key)
    if cached is not None:
        return RecommendationResponse.model_validate_json(cached)

    response = recommend_for_user(
        db,
        user,
        search_client=search_client,
        embedding_model=embedding_model,
        ranker=ranker,
        model_version=model_version,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        vector_client=vector_client,
    )
    RECOMMENDATIONS_SERVED.labels(
        model_version=model_version,
        strategy=response.strategy,
    ).inc()
    cache_set(cache_client, cache_key, response.model_dump_json(), cache_ttl_seconds())
    return response
