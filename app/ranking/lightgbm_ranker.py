from pathlib import Path

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
    dataframe["article_age_hours"] = dataframe["article_age_hours"].fillna(0.0)
    return dataframe.sort_values(GROUP_COLUMN).reset_index(drop=True)

# convert sorted rows into LightGBM group sizes
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

def train_ranker(dataframe: pd.DataFrame) -> lgb.LGBMRanker:
    features, labels, group_sizes = split_features_labels_groups(dataframe)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    ranker.fit(
        features,
        labels,
        group=group_sizes,
    )

    return ranker

# evaluate the ranker using NDCG@10
def evaluate_ranker(
    model: lgb.LGBMRanker,
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    sorted_dataframe = dataframe.sort_values(GROUP_COLUMN).reset_index(drop=True)
    features = sorted_dataframe[FEATURE_COLUMNS]

    predictions = model.predict(features)
    evaluation_frame = sorted_dataframe[[GROUP_COLUMN, LABEL_COLUMN]].copy()
    evaluation_frame["prediction"] = predictions

    ndcg_scores = []
    for _, group in evaluation_frame.groupby(GROUP_COLUMN, sort=False):
        if group[LABEL_COLUMN].nunique() < 2:
            continue

        y_true = [group[LABEL_COLUMN].to_list()]
        y_score = [group["prediction"].to_list()]
        ndcg_scores.append(ndcg_score(y_true, y_score, k=10))

    if not ndcg_scores:
        return {"ndcg_at_10": 0.0}

    return {"ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores)}

def save_model(model: lgb.LGBMRanker, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(path)

def load_model(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))