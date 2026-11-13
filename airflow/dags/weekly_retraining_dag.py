"""Weekly retraining DAG: rebuild features, retrain, evaluate, and promote."""

import pendulum
from common import DATA_MOUNT_PATH, DEFAULT_ARGS, pipeline_task

from airflow import DAG

DATASET_PATH = f"{DATA_MOUNT_PATH}/ranking/dataset.parquet"
MODEL_PATH = f"{DATA_MOUNT_PATH}/models/ranker.txt"
METRICS_PATH = f"{DATA_MOUNT_PATH}/reports/eval_metrics.json"

MIN_NDCG = "0.5"

with DAG(
    dag_id="weekly_retraining",
    description="Retrain the LightGBM ranker from logged feedback and promote it",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["news-rec", "training"],
) as dag:
    build_training_dataset = pipeline_task(
        task_id="build_training_dataset",
        module="build_ranking_dataset",
        arguments=["--output-path", DATASET_PATH, "--limit", "50000"],
    )

    train_ranker = pipeline_task(
        task_id="train_ranker",
        module="train_ranker",
        arguments=[
            "--input-path",
            DATASET_PATH,
            "--output-path",
            MODEL_PATH,
            "--register-model",
        ],
    )

    evaluate_model = pipeline_task(
        task_id="evaluate_model",
        module="evaluate_model",
        arguments=[
            "--model-path",
            MODEL_PATH,
            "--dataset-path",
            DATASET_PATH,
            "--output-path",
            METRICS_PATH,
            "--min-ndcg",
            MIN_NDCG,
        ],
    )

    promote_model = pipeline_task(
        task_id="promote_model",
        module="promote_model",
        arguments=[
            "--metrics-path",
            METRICS_PATH,
            "--record-db",
        ],
    )

    build_training_dataset >> train_ranker >> evaluate_model >> promote_model
