import argparse
import os
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.models import infer_signature

from app.ranking.lightgbm_ranker import (
    DEFAULT_RANKER_PARAMS,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    evaluate_ranker,
    load_training_data,
    save_model,
    train_ranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM learning-to-rank model.")
    parser.add_argument(
        "--input-path",
        required=True,
        help="Parquet file from build_ranking_dataset.py",
    )
    parser.add_argument("--output-path", required=True, help="Where to save the trained model")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        help="MLflow tracking URI (default: file:./mlruns or MLFLOW_TRACKING_URI)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="news-ranker",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the trained model in the MLflow Model Registry",
    )
    parser.add_argument(
        "--model-name",
        default="news-lightgbm-ranker",
        help="Registered model name when --register-model is set",
    )
    return parser.parse_args()


def split_train_validation_groups(
    dataframe: pd.DataFrame,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_fraction < 1.0:
        msg = "validation_fraction must be between 0 and 1"
        raise ValueError(msg)

    unique_groups = dataframe[GROUP_COLUMN].drop_duplicates()
    if len(unique_groups) < 2:
        msg = "Need at least 2 impression groups to split train and validation data"
        raise ValueError(msg)

    validation_count = max(1, int(len(unique_groups) * validation_fraction))
    if validation_count >= len(unique_groups):
        validation_count = len(unique_groups) - 1

    validation_groups = unique_groups.sample(n=validation_count, random_state=random_state)
    validation_mask = dataframe[GROUP_COLUMN].isin(validation_groups)

    train = dataframe.loc[~validation_mask].reset_index(drop=True)
    validation = dataframe.loc[validation_mask].reset_index(drop=True)
    return train, validation


def run(
    input_path: Path,
    output_path: Path,
    validation_fraction: float,
    random_state: int,
    *,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    register_model: bool,
    model_name: str,
) -> dict[str, float]:
    dataframe = load_training_data(input_path)
    if dataframe.empty:
        msg = f"No training rows found in {input_path}"
        raise ValueError(msg)

    train, validation = split_train_validation_groups(
        dataframe,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="train-lightgbm-ranker") as run:
        mlflow.log_params(
            {
                **{f"ranker_{key}": value for key, value in DEFAULT_RANKER_PARAMS.items()},
                "validation_fraction": validation_fraction,
                "random_state": random_state,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_groups": int(train[GROUP_COLUMN].nunique()),
                "validation_groups": int(validation[GROUP_COLUMN].nunique()),
                "input_path": str(input_path),
            }
        )

        model = train_ranker(train)
        metrics = evaluate_ranker(model, validation)
        save_model(model, output_path)

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_path), artifact_path="model")

        signature = infer_signature(
            train[FEATURE_COLUMNS],
            model.predict(train[FEATURE_COLUMNS]),
        )
        model_info = mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="lightgbm_ranker",
            signature=signature,
            registered_model_name=model_name if register_model else None,
        )

        mlflow.set_tags(
            {
                "pipeline": "train_ranker",
                "framework": "lightgbm",
                "task": "learning-to-rank",
            }
        )

        print(f"MLflow run id: {run.info.run_id}")
        print(f"MLflow model URI: {model_info.model_uri}")

    return metrics


def main() -> None:
    args = parse_args()
    metrics = run(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        register_model=args.register_model,
        model_name=args.model_name,
    )
    print(f"Saved ranker to {args.output_path}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
