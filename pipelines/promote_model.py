import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from app.db.models import ModelVersion
from app.db.session import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote the latest registered model version in MLflow."
    )
    parser.add_argument("--model-name", default="news-lightgbm-ranker")
    parser.add_argument("--alias", default="production")
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Metrics JSON from evaluate_model.py, recorded alongside the promotion",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    )
    parser.add_argument(
        "--record-db",
        action="store_true",
        help="Also record the promotion in the model_versions table",
    )
    return parser.parse_args()


def latest_model_version(client: MlflowClient, model_name: str) -> Any:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        msg = f"No registered versions found for model {model_name!r}"
        raise ValueError(msg)
    return max(versions, key=lambda version: int(version.version))


def record_promotion_in_db(
    model_name: str,
    version: str,
    artifact_uri: str | None,
    metrics: dict[str, Any] | None,
) -> None:
    db = SessionLocal()
    try:
        version_key = f"{model_name}/{version}"
        existing = (
            db.query(ModelVersion).filter(ModelVersion.version == version_key).one_or_none()
        )
        if existing is None:
            db.add(
                ModelVersion(
                    name=model_name,
                    version=version_key,
                    status="production",
                    artifact_uri=artifact_uri,
                    metrics=metrics,
                )
            )
        else:
            existing.status = "production"
            if metrics is not None:
                existing.metrics = metrics
        # Only one version should be marked production at a time.
        db.query(ModelVersion).filter(
            ModelVersion.name == model_name,
            ModelVersion.version != version_key,
            ModelVersion.status == "production",
        ).update({"status": "archived"})
        db.commit()
    finally:
        db.close()


def run(
    model_name: str,
    alias: str,
    *,
    mlflow_tracking_uri: str,
    metrics_path: Path | None = None,
    record_db: bool = False,
) -> str:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    version = latest_model_version(client, model_name)
    client.set_registered_model_alias(model_name, alias, version.version)

    metrics: dict[str, Any] | None = None
    if metrics_path is not None and metrics_path.exists():
        report = json.loads(metrics_path.read_text())
        metrics = report.get("metrics")

    if record_db:
        record_promotion_in_db(
            model_name=model_name,
            version=version.version,
            artifact_uri=version.source,
            metrics=metrics,
        )

    return str(version.version)


def main() -> None:
    args = parse_args()
    version = run(
        model_name=args.model_name,
        alias=args.alias,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        metrics_path=Path(args.metrics_path) if args.metrics_path else None,
        record_db=args.record_db,
    )
    print(f"Promoted {args.model_name} version {version} to alias {args.alias!r}")


if __name__ == "__main__":
    main()
