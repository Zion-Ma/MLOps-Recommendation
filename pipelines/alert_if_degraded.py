import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check monitoring reports and fail when the system is degraded."
    )
    parser.add_argument("--ctr-path", default=None, help="CTR JSON from compute_ctr.py")
    parser.add_argument(
        "--latency-path", default=None, help="Latency JSON from compute_latency.py"
    )
    parser.add_argument(
        "--drift-summary-path",
        default=None,
        help="Drift summary JSON from generate_drift_report.py",
    )
    parser.add_argument("--output-path", required=True, help="Where to write the alert JSON")
    parser.add_argument("--min-ctr", type=float, default=0.01)
    parser.add_argument("--max-latency-seconds", type=float, default=1.0)
    parser.add_argument("--max-error-rate", type=float, default=0.05)
    parser.add_argument("--max-drift-share", type=float, default=0.5)
    return parser.parse_args()


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text()))


def check_ctr(report: dict[str, Any] | None, min_ctr: float) -> list[str]:
    if report is None:
        return []
    ctr = report.get("ctr")
    if ctr is None:
        # No impressions in the window is worth surfacing but is not degradation.
        return []
    if float(ctr) < min_ctr:
        return [f"CTR {float(ctr):.4f} below threshold {min_ctr:.4f}"]
    return []


def check_latency(
    report: dict[str, Any] | None,
    max_latency_seconds: float,
    max_error_rate: float,
) -> list[str]:
    if report is None:
        return []
    violations = []

    latency = report.get("latency_seconds")
    if latency is not None and float(latency) > max_latency_seconds:
        violations.append(
            f"p95 latency {float(latency):.3f}s above threshold {max_latency_seconds:.3f}s"
        )

    error_rate = report.get("error_rate")
    if error_rate is not None and float(error_rate) > max_error_rate:
        violations.append(
            f"Error rate {float(error_rate):.4f} above threshold {max_error_rate:.4f}"
        )
    return violations


def check_drift(report: dict[str, Any] | None, max_drift_share: float) -> list[str]:
    if report is None:
        return []
    drift_share = float(report.get("drift_share", 0.0))
    if drift_share > max_drift_share:
        drifted = ", ".join(report.get("drifted_features", []))
        return [
            f"Feature drift share {drift_share:.2f} above threshold {max_drift_share:.2f} "
            f"(drifted: {drifted})"
        ]
    return []


def run(
    output_path: Path,
    *,
    ctr_path: Path | None = None,
    latency_path: Path | None = None,
    drift_summary_path: Path | None = None,
    min_ctr: float = 0.01,
    max_latency_seconds: float = 1.0,
    max_error_rate: float = 0.05,
    max_drift_share: float = 0.5,
) -> dict[str, Any]:
    violations: list[str] = []
    violations += check_ctr(_load_json(ctr_path), min_ctr)
    violations += check_latency(_load_json(latency_path), max_latency_seconds, max_error_rate)
    violations += check_drift(_load_json(drift_summary_path), max_drift_share)

    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "degraded": bool(violations),
        "violations": violations,
        "thresholds": {
            "min_ctr": min_ctr,
            "max_latency_seconds": max_latency_seconds,
            "max_error_rate": max_error_rate,
            "max_drift_share": max_drift_share,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parse_args()
    report = run(
        output_path=Path(args.output_path),
        ctr_path=Path(args.ctr_path) if args.ctr_path else None,
        latency_path=Path(args.latency_path) if args.latency_path else None,
        drift_summary_path=Path(args.drift_summary_path) if args.drift_summary_path else None,
        min_ctr=args.min_ctr,
        max_latency_seconds=args.max_latency_seconds,
        max_error_rate=args.max_error_rate,
        max_drift_share=args.max_drift_share,
    )
    if report["degraded"]:
        for violation in report["violations"]:
            print(f"ALERT: {violation}")
        # A non-zero exit fails the Airflow task, which is the alerting signal.
        raise SystemExit(1)
    print("System healthy: no monitoring thresholds violated")


if __name__ == "__main__":
    main()
