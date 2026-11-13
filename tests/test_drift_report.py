import numpy as np
import pandas as pd

from pipelines.generate_drift_report import (
    build_drift_summary,
    population_stability_index,
    run,
)

RNG = np.random.default_rng(42)


def make_features(shift=0.0, rows=500):
    return pd.DataFrame(
        {
            "bm25_score": RNG.normal(10 + shift * 10, 2, rows),
            "dense_score": RNG.uniform(0, 1, rows) + shift,
            "rerank_score": RNG.uniform(0, 1, rows) + shift,
            "article_age_hours": RNG.exponential(24, rows),
            "article_popularity": RNG.poisson(2, rows).astype(float),
            "user_category_affinity": RNG.uniform(0, 1, rows),
        }
    )


def test_psi_is_near_zero_for_same_distribution():
    values = pd.Series(RNG.normal(0, 1, 2000))
    other = pd.Series(RNG.normal(0, 1, 2000))

    assert population_stability_index(values, other) < 0.1


def test_psi_detects_shifted_distribution():
    reference = pd.Series(RNG.normal(0, 1, 2000))
    current = pd.Series(RNG.normal(3, 1, 2000))

    assert population_stability_index(reference, current) > 0.2


def test_psi_handles_empty_series():
    assert population_stability_index(pd.Series(dtype=float), pd.Series([1.0])) == 0.0


def test_build_drift_summary_flags_drifted_features():
    reference = make_features()
    current = make_features(shift=5.0)

    summary = build_drift_summary(
        reference,
        current,
        columns=["bm25_score", "article_age_hours"],
        psi_threshold=0.2,
    )

    assert summary["features"]["bm25_score"]["drifted"] is True
    assert "bm25_score" in summary["drifted_features"]
    assert 0.0 <= summary["drift_share"] <= 1.0


def test_run_writes_summary_and_evidently_report(tmp_path):
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    make_features().to_parquet(reference_path, index=False)
    make_features(shift=2.0).to_parquet(current_path, index=False)

    output_dir = tmp_path / "reports"
    summary = run(reference_path, current_path, output_dir)

    assert (output_dir / "drift_summary.json").exists()
    assert (output_dir / "drift_report.html").exists()
    assert summary["drifted_features"]
