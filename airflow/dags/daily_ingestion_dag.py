"""Daily ingestion DAG: pull new Hacker News stories and refresh both indexes."""

import pendulum
from common import DEFAULT_ARGS, pipeline_task

from airflow import DAG

with DAG(
    dag_id="daily_ingestion",
    description="Ingest Hacker News articles and update OpenSearch and Qdrant",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["news-rec", "ingestion"],
) as dag:
    fetch_hn_articles = pipeline_task(
        task_id="fetch_hn_articles",
        module="ingest_hn",
        arguments=["--days-back", "1", "--limit", "500"],
        mount_data=False,
    )

    update_opensearch = pipeline_task(
        task_id="update_opensearch",
        module="index_bm25",
        arguments=["--limit", "1000"],
        mount_data=False,
    )

    update_qdrant = pipeline_task(
        task_id="update_qdrant",
        module="index_dense",
        arguments=["--limit", "1000"],
        mount_data=False,
    )

    fetch_hn_articles >> update_opensearch >> update_qdrant
