"""Shared helpers for news-rec DAGs.

Every pipeline step runs as a KubernetesPodOperator using the project image, so
Airflow only orchestrates; the work happens in isolated pods that read their
connection settings from the news-rec ConfigMap and Secret.
"""

import os
from typing import Any

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

IMAGE = os.getenv("NEWS_REC_IMAGE", "ghcr.io/zion-ma/mlops-recommendation:latest")
NAMESPACE = os.getenv("NEWS_REC_NAMESPACE", "news-rec")
CONFIGMAP_NAME = os.getenv("NEWS_REC_CONFIGMAP", "news-rec-config")
SECRET_NAME = os.getenv("NEWS_REC_SECRET", "news-rec-secrets")
DATA_PVC_NAME = os.getenv("NEWS_REC_DATA_PVC", "news-rec-data-pvc")

# Shared PVC where tasks exchange datasets, models, and reports.
DATA_MOUNT_PATH = "/data"

DATA_VOLUME = k8s.V1Volume(
    name="news-rec-data",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name=DATA_PVC_NAME,
    ),
)
DATA_VOLUME_MOUNT = k8s.V1VolumeMount(
    name="news-rec-data",
    mount_path=DATA_MOUNT_PATH,
)

ENV_FROM = [
    k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name=CONFIGMAP_NAME)),
    k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name=SECRET_NAME)),
]

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "mlops",
    "retries": 1,
}


def pipeline_task(
    task_id: str,
    module: str,
    arguments: list[str] | None = None,
    *,
    mount_data: bool = True,
) -> KubernetesPodOperator:
    """Run `python -m pipelines.<module>` in a pod built from the project image."""
    return KubernetesPodOperator(
        task_id=task_id,
        name=task_id.replace("_", "-"),
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=["python", "-m", f"pipelines.{module}"],
        arguments=arguments or [],
        env_from=ENV_FROM,
        volumes=[DATA_VOLUME] if mount_data else [],
        volume_mounts=[DATA_VOLUME_MOUNT] if mount_data else [],
        get_logs=True,
        on_finish_action="delete_pod",
    )
