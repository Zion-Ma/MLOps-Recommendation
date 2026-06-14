from pathlib import Path

import pandas as pd
import pytest

from app.ranking.lightgbm_ranker import (
    FEATURE_COLUMNS,
    build_group_sizes,
    evaluate_ranker,
    load_training_data,
    split_features_labels_groups,
)


def make_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": ["b", "a", "a", "b", "b", "c"],
            "label": [0, 1, 0, 1, 0, 0],
            "bm25_score": [0.1, 0.8, 0.2, 0.9, 0.3, 0.4],
            "dense_score": [0.2, 0.7, 0.1, 0.8, 0.4, 0.5],
            "rerank_score": [0.2, 0.7, 0.1, 0.8, 0.4, 0.5],
            "article_age_hours": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            "article_popularity": [1.0, 2.0, 0.0, 3.0, 1.0, 0.0],
            "user_category_affinity": [0.0, 0.5, 0.1, 0.7, 0.2, 0.3],
        }
    )


def test_build_group_sizes_preserves_sorted_group_order() -> None:
    dataframe = pd.DataFrame({"group_id": ["a", "a", "b", "b", "b", "c"]})

    assert build_group_sizes(dataframe) == [2, 3, 1]


def test_split_features_labels_groups_returns_model_inputs() -> None:
    dataframe = make_training_frame()

    features, labels, group_sizes = split_features_labels_groups(dataframe)

    assert list(features.columns) == FEATURE_COLUMNS
    assert labels.to_list() == [1, 0, 0, 1, 0, 0]
    assert group_sizes == [2, 3, 1]


def test_load_training_data_sorts_groups_and_fills_article_age(tmp_path: Path) -> None:
    path = tmp_path / "ranking_features.parquet"
    make_training_frame().to_parquet(path, index=False)

    dataframe = load_training_data(path)

    assert dataframe["group_id"].to_list() == ["a", "a", "b", "b", "b", "c"]
    assert dataframe["article_age_hours"].to_list()[0] == 0.0


def test_load_training_data_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_features.parquet"
    pd.DataFrame({"group_id": ["a"], "label": [1]}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_training_data(path)


class FakeRanker:
    def predict(self, features: pd.DataFrame) -> list[float]:
        return [0.2, 0.9, 0.1, 0.8]


def test_evaluate_ranker_returns_ndcg_at_10() -> None:
    dataframe = pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "b"],
            "label": [0, 1, 0, 1],
            "bm25_score": [0.1, 0.8, 0.2, 0.9],
            "dense_score": [0.2, 0.7, 0.3, 0.8],
            "rerank_score": [0.2, 0.7, 0.3, 0.8],
            "article_age_hours": [1.0, 2.0, 3.0, 4.0],
            "article_popularity": [0.0, 1.0, 0.0, 1.0],
            "user_category_affinity": [0.0, 0.5, 0.1, 0.7],
        }
    )

    metrics = evaluate_ranker(FakeRanker(), dataframe)

    assert metrics["ndcg_at_10"] == 1.0


def test_evaluate_ranker_returns_zero_without_mixed_label_groups() -> None:
    dataframe = pd.DataFrame(
        {
            "group_id": ["a", "a"],
            "label": [0, 0],
            "bm25_score": [0.1, 0.8],
            "dense_score": [0.2, 0.7],
            "rerank_score": [0.2, 0.7],
            "article_age_hours": [1.0, 2.0],
            "article_popularity": [0.0, 1.0],
            "user_category_affinity": [0.0, 0.5],
        }
    )

    metrics = evaluate_ranker(FakeRanker(), dataframe)

    assert metrics == {"ndcg_at_10": 0.0}
