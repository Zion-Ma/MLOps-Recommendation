import sys
from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow", reason="apache-airflow not installed")
pytest.importorskip(
    "airflow.providers.cncf.kubernetes",
    reason="cncf.kubernetes provider not installed",
)

DAGS_FOLDER = Path(__file__).resolve().parents[1] / "airflow" / "dags"

EXPECTED_DAGS = {
    "daily_ingestion": ["fetch_hn_articles", "update_opensearch", "update_qdrant"],
    "weekly_retraining": [
        "build_training_dataset",
        "train_ranker",
        "evaluate_model",
        "promote_model",
    ],
    "monitoring": [
        "compute_ctr",
        "compute_latency",
        "build_current_features",
        "generate_drift_report",
        "alert_if_degraded",
    ],
}


@pytest.fixture(scope="module")
def dag_bag():
    from airflow.models import DagBag

    # Airflow normally puts the dags folder on sys.path; mirror that here so
    # `from common import ...` resolves during the import check.
    sys.path.insert(0, str(DAGS_FOLDER))
    try:
        yield DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    finally:
        sys.path.remove(str(DAGS_FOLDER))


def test_dags_import_without_errors(dag_bag):
    assert dag_bag.import_errors == {}


def test_expected_dags_present(dag_bag):
    assert set(EXPECTED_DAGS) <= set(dag_bag.dag_ids)


@pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAGS))
def test_dag_tasks_and_schedule(dag_bag, dag_id):
    dag = dag_bag.dags[dag_id]

    assert dag is not None
    assert sorted(task.task_id for task in dag.tasks) == sorted(EXPECTED_DAGS[dag_id])
    assert dag.catchup is False


def test_retraining_dag_task_order(dag_bag):
    dag = dag_bag.dags["weekly_retraining"]

    train = dag.get_task("train_ranker")
    assert {task.task_id for task in train.upstream_list} == {"build_training_dataset"}
    assert {task.task_id for task in train.downstream_list} == {"evaluate_model"}


def test_monitoring_dag_alert_depends_on_all_checks(dag_bag):
    dag = dag_bag.dags["monitoring"]

    alert = dag.get_task("alert_if_degraded")
    assert {task.task_id for task in alert.upstream_list} == {
        "compute_ctr",
        "compute_latency",
        "generate_drift_report",
    }
