"""Monitoring DAG: compute CTR and latency, generate drift reports, and alert."""

import pendulum
from common import DATA_MOUNT_PATH, DEFAULT_ARGS, pipeline_task

from airflow import DAG

CTR_PATH = f"{DATA_MOUNT_PATH}/reports/ctr.json"
LATENCY_PATH = f"{DATA_MOUNT_PATH}/reports/latency.json"
DRIFT_DIR = f"{DATA_MOUNT_PATH}/reports/drift"
ALERTS_PATH = f"{DATA_MOUNT_PATH}/reports/alerts.json"
REFERENCE_DATASET = f"{DATA_MOUNT_PATH}/ranking/dataset.parquet"
CURRENT_DATASET = f"{DATA_MOUNT_PATH}/ranking/current.parquet"

with DAG(
    dag_id="monitoring",
    description="Daily system and model health checks with drift detection",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["news-rec", "monitoring"],
) as dag:
    compute_ctr = pipeline_task(
        task_id="compute_ctr",
        module="compute_ctr",
        arguments=["--window-hours", "24", "--output-path", CTR_PATH],
    )

    compute_latency = pipeline_task(
        task_id="compute_latency",
        module="compute_latency",
        arguments=["--window", "1h", "--output-path", LATENCY_PATH],
    )

    build_current_features = pipeline_task(
        task_id="build_current_features",
        module="build_ranking_dataset",
        arguments=["--output-path", CURRENT_DATASET, "--limit", "10000"],
    )

    generate_drift_report = pipeline_task(
        task_id="generate_drift_report",
        module="generate_drift_report",
        arguments=[
            "--reference-path",
            REFERENCE_DATASET,
            "--current-path",
            CURRENT_DATASET,
            "--output-dir",
            DRIFT_DIR,
        ],
    )

    alert_if_degraded = pipeline_task(
        task_id="alert_if_degraded",
        module="alert_if_degraded",
        arguments=[
            "--ctr-path",
            CTR_PATH,
            "--latency-path",
            LATENCY_PATH,
            "--drift-summary-path",
            f"{DRIFT_DIR}/drift_summary.json",
            "--output-path",
            ALERTS_PATH,
        ],
    )

    build_current_features >> generate_drift_report
    [compute_ctr, compute_latency, generate_drift_report] >> alert_if_degraded
