import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.ranking.lightgbm_ranker import evaluate_ranker, load_model, load_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LightGBM ranker.")
    parser.add_argument("--model-path", required=True, help="Saved LightGBM model file")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Parquet evaluation dataset from build_ranking_dataset.py",
    )
    parser.add_argument("--output-path", required=True, help="Where to write metrics JSON")
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--recall-k", type=int, default=50)
    parser.add_argument(
        "--min-ndcg",
        type=float,
        default=None,
        help="Fail (exit 1) when NDCG@k falls below this threshold",
    )
    return parser.parse_args()


def run(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    ndcg_k: int = 10,
    recall_k: int = 50,
    min_ndcg: float | None = None,
) -> dict[str, float]:
    dataframe = load_training_data(dataset_path)
    if dataframe.empty:
        msg = f"No evaluation rows found in {dataset_path}"
        raise ValueError(msg)

    model = load_model(model_path)
    metrics = evaluate_ranker(model, dataframe, ndcg_k=ndcg_k, recall_k=recall_k)

    report = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "rows": len(dataframe),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    if min_ndcg is not None and metrics["ndcg_at_10"] < min_ndcg:
        msg = (
            f"Model failed evaluation gate: NDCG@10 {metrics['ndcg_at_10']:.4f} "
            f"< threshold {min_ndcg:.4f}"
        )
        raise SystemExit(msg)

    return metrics


def main() -> None:
    args = parse_args()
    metrics = run(
        model_path=Path(args.model_path),
        dataset_path=Path(args.dataset_path),
        output_path=Path(args.output_path),
        ndcg_k=args.ndcg_k,
        recall_k=args.recall_k,
        min_ndcg=args.min_ndcg,
    )
    print(f"Wrote evaluation metrics to {args.output_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
