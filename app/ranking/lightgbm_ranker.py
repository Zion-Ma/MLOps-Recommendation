from pathlib import Path
from typing import Any, Protocol

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import ndcg_score

FEATURE_COLUMNS = [
    "bm25_score",
    "dense_score",
    "rerank_score",
    "article_age_hours",
    "article_popularity",
    "user_category_affinity",
]
LABEL_COLUMN = "label"
GROUP_COLUMN = "group_id"

DEFAULT_RANKER_PARAMS: dict[str, float | int | str] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
}


class RankerPredict(Protocol):
    # Positional-only so both LGBMRanker (X) and Booster (data) conform.
    def predict(self, features: pd.DataFrame, /) -> Any: ...


def load_training_data(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)
    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN, GROUP_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    dataframe = dataframe.dropna(subset=[LABEL_COLUMN, GROUP_COLUMN])
    dataframe = dataframe.copy()
    dataframe["article_age_hours"] = pd.to_numeric(
        dataframe["article_age_hours"], errors="coerce"
    ).fillna(0.0)
    return dataframe.sort_values(GROUP_COLUMN).reset_index(drop=True)


def build_group_sizes(dataframe: pd.DataFrame) -> list[int]:
    group_sizes = dataframe.groupby(GROUP_COLUMN, sort=False).size().tolist()
    return [int(size) for size in group_sizes]


def split_features_labels_groups(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    sorted_dataframe = dataframe.sort_values(GROUP_COLUMN).reset_index(drop=True)

    features = sorted_dataframe[FEATURE_COLUMNS]
    labels = sorted_dataframe[LABEL_COLUMN].astype(int)
    group_sizes = build_group_sizes(sorted_dataframe)

    return features, labels, group_sizes


def train_ranker(
    dataframe: pd.DataFrame,
    params: dict[str, float | int | str] | None = None,
) -> lgb.LGBMRanker:
    features, labels, group_sizes = split_features_labels_groups(dataframe)
    ranker_params = {**DEFAULT_RANKER_PARAMS, **(params or {})}

    ranker = lgb.LGBMRanker(
        objective=str(ranker_params["objective"]),
        metric=str(ranker_params["metric"]),
        n_estimators=int(ranker_params["n_estimators"]),
        learning_rate=float(ranker_params["learning_rate"]),
        num_leaves=int(ranker_params["num_leaves"]),
        random_state=int(ranker_params["random_state"]),
    )

    ranker.fit(
        features,
        labels,
        group=group_sizes,
    )

    return ranker


def _group_ndcg_at_k(labels: list[int], scores: list[float], k: int) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(ndcg_score([labels], [scores], k=k))


def _group_mrr(labels: list[int], scores: list[float]) -> float | None:
    if not any(label > 0 for label in labels):
        return None

    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    for rank, (_, label) in enumerate(ranked, start=1):
        if label > 0:
            return 1.0 / rank
    return 0.0


def _group_recall_at_k(labels: list[int], scores: list[float], k: int) -> float | None:
    relevant = sum(1 for label in labels if label > 0)
    if relevant == 0:
        return None

    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    hits = sum(1 for _, label in ranked[:k] if label > 0)
    return hits / relevant


def evaluate_ranker(
    model: RankerPredict,
    dataframe: pd.DataFrame,
    *,
    ndcg_k: int = 10,
    recall_k: int = 50,
) -> dict[str, float]:
    sorted_dataframe = dataframe.sort_values(GROUP_COLUMN).reset_index(drop=True)
    features = sorted_dataframe[FEATURE_COLUMNS]

    predictions = list(model.predict(features))
    evaluation_frame = sorted_dataframe[[GROUP_COLUMN, LABEL_COLUMN]].copy()
    evaluation_frame["prediction"] = predictions

    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []
    recall_scores: list[float] = []

    for _, group in evaluation_frame.groupby(GROUP_COLUMN, sort=False):
        labels = group[LABEL_COLUMN].astype(int).tolist()
        scores = group["prediction"].astype(float).tolist()

        ndcg = _group_ndcg_at_k(labels, scores, k=ndcg_k)
        if ndcg is not None:
            ndcg_scores.append(ndcg)

        mrr = _group_mrr(labels, scores)
        if mrr is not None:
            mrr_scores.append(mrr)

        recall = _group_recall_at_k(labels, scores, k=recall_k)
        if recall is not None:
            recall_scores.append(recall)

    return {
        "ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "recall_at_50": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
    }


def save_model(model: lgb.LGBMRanker, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def load_model(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
