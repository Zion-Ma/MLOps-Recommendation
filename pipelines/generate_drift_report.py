import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ranking.lightgbm_ranker import FEATURE_COLUMNS

PSI_DRIFT_THRESHOLD = 0.2
PSI_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Evidently data drift report for ranking features."
    )
    parser.add_argument("--reference-path", required=True, help="Reference dataset (parquet)")
    parser.add_argument("--current-path", required=True, help="Current dataset (parquet)")
    parser.add_argument("--output-dir", required=True, help="Directory for report files")
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=PSI_DRIFT_THRESHOLD,
        help="PSI above this value marks a feature as drifted",
    )
    return parser.parse_args()


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = PSI_BINS,
) -> float:
    """PSI between two numeric distributions using reference quantile bins."""
    reference_values = reference.dropna().to_numpy(dtype=float)
    current_values = current.dropna().to_numpy(dtype=float)
    if len(reference_values) == 0 or len(current_values) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if len(edges) < 2:
        # Constant reference feature: any new value is full drift.
        return 0.0 if np.array_equal(np.unique(current_values), edges) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)

    # Small epsilon avoids division by zero for empty bins.
    reference_fractions = np.clip(reference_counts / len(reference_values), 1e-6, None)
    current_fractions = np.clip(current_counts / len(current_values), 1e-6, None)

    psi = np.sum(
        (current_fractions - reference_fractions)
        * np.log(current_fractions / reference_fractions)
    )
    return float(psi)


def build_drift_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    psi_threshold: float,
) -> dict[str, Any]:
    features: dict[str, dict[str, Any]] = {}
    for column in columns:
        psi = population_stability_index(reference[column], current[column])
        features[column] = {
            "psi": psi,
            "drifted": bool(psi > psi_threshold),
        }

    drifted = [name for name, result in features.items() if result["drifted"]]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "psi_threshold": psi_threshold,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "features": features,
        "drifted_features": drifted,
        "drift_share": len(drifted) / len(columns) if columns else 0.0,
    }


def generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write an Evidently data drift HTML report, supporting old and new APIs."""
    try:  # evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        snapshot = report.run(current_data=current, reference_data=reference)
        snapshot.save_html(str(output_path))
    except ImportError:  # evidently < 0.7
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        legacy_report = Report(metrics=[DataDriftPreset()])
        legacy_report.run(reference_data=reference, current_data=current)
        legacy_report.save_html(str(output_path))


def run(
    reference_path: Path,
    current_path: Path,
    output_dir: Path,
    psi_threshold: float = PSI_DRIFT_THRESHOLD,
) -> dict[str, Any]:
    reference = pd.read_parquet(reference_path)
    current = pd.read_parquet(current_path)

    columns = [
        column
        for column in FEATURE_COLUMNS
        if column in reference.columns and column in current.columns
    ]
    if not columns:
        msg = "No shared ranking feature columns between reference and current datasets"
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_drift_summary(reference, current, columns, psi_threshold)
    summary_path = output_dir / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    generate_evidently_report(
        reference[columns],
        current[columns],
        output_dir / "drift_report.html",
    )

    return summary


def main() -> None:
    args = parse_args()
    summary = run(
        reference_path=Path(args.reference_path),
        current_path=Path(args.current_path),
        output_dir=Path(args.output_dir),
        psi_threshold=args.psi_threshold,
    )
    print(f"Wrote drift report to {args.output_dir}")
    print(
        f"Drifted features: {summary['drifted_features'] or 'none'} "
        f"(share {summary['drift_share']:.2f})"
    )


if __name__ == "__main__":
    main()
