import json

import pandas as pd
import pytest

from app.ranking.lightgbm_ranker import save_model, train_ranker
from pipelines.evaluate_model import run


def make_dataset(groups=8, rows_per_group=6):
    records = []
    for group in range(groups):
        for row in range(rows_per_group):
            records.append(
                {
                    "group_id": f"group-{group}",
                    "bm25_score": float(row),
                    "dense_score": float(row) / rows_per_group,
                    "rerank_score": float(row) / rows_per_group,
                    "article_age_hours": 1.0 + row,
                    "article_popularity": float(row % 3),
                    "user_category_affinity": 0.5,
                    "label": 1 if row == rows_per_group - 1 else 0,
                }
            )
    return pd.DataFrame.from_records(records)


@pytest.fixture
def trained_model_path(tmp_path):
    dataframe = make_dataset()
    model = train_ranker(dataframe)
    model_path = tmp_path / "ranker.txt"
    save_model(model, model_path)
    dataset_path = tmp_path / "dataset.parquet"
    dataframe.to_parquet(dataset_path, index=False)
    return model_path, dataset_path


def test_evaluate_model_writes_metrics_json(trained_model_path, tmp_path):
    model_path, dataset_path = trained_model_path
    output_path = tmp_path / "metrics.json"

    metrics = run(model_path, dataset_path, output_path)

    assert set(metrics) == {"ndcg_at_10", "mrr", "recall_at_50"}
    report = json.loads(output_path.read_text())
    assert report["metrics"] == metrics
    assert report["rows"] > 0


def test_evaluate_model_gate_fails_below_threshold(trained_model_path, tmp_path):
    model_path, dataset_path = trained_model_path
    output_path = tmp_path / "metrics.json"

    with pytest.raises(SystemExit):
        run(model_path, dataset_path, output_path, min_ndcg=2.0)


def test_evaluate_model_rejects_empty_dataset(trained_model_path, tmp_path):
    model_path, _ = trained_model_path
    empty_path = tmp_path / "empty.parquet"
    make_dataset(groups=0).reindex(
        columns=[
            "group_id",
            "bm25_score",
            "dense_score",
            "rerank_score",
            "article_age_hours",
            "article_popularity",
            "user_category_affinity",
            "label",
        ]
    ).to_parquet(empty_path, index=False)

    with pytest.raises(ValueError):
        run(model_path, empty_path, tmp_path / "metrics.json")
