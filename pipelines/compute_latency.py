import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

LATENCY_QUERY_TEMPLATE = (
    "histogram_quantile({quantile}, "
    "sum(rate(http_request_latency_seconds_bucket[{window}])) by (le))"
)
ERROR_RATE_QUERY_TEMPLATE = (
    'sum(rate(http_requests_total{{status=~"5.."}}[{window}])) '
    "/ sum(rate(http_requests_total[{window}]))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch API latency and error rate from Prometheus."
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--window", default="1h", help="Prometheus range window, e.g. 5m or 1h")
    parser.add_argument("--output-path", required=True, help="Where to write the latency JSON")
    return parser.parse_args()


def query_prometheus(client: httpx.Client, prometheus_url: str, query: str) -> float | None:
    response = client.get(f"{prometheus_url}/api/v1/query", params={"query": query})
    response.raise_for_status()
    payload = response.json()

    results = payload.get("data", {}).get("result", [])
    if not results:
        return None
    value = float(results[0]["value"][1])
    # Prometheus returns NaN when there is no data in the window.
    if value != value:
        return None
    return value


def run(
    prometheus_url: str,
    output_path: Path,
    *,
    quantile: float = 0.95,
    window: str = "1h",
) -> dict[str, Any]:
    latency_query = LATENCY_QUERY_TEMPLATE.format(quantile=quantile, window=window)
    error_rate_query = ERROR_RATE_QUERY_TEMPLATE.format(window=window)

    with httpx.Client(timeout=10.0) as client:
        latency = query_prometheus(client, prometheus_url, latency_query)
        error_rate = query_prometheus(client, prometheus_url, error_rate_query)

    report = {
        "prometheus_url": prometheus_url,
        "quantile": quantile,
        "window": window,
        "computed_at": datetime.now(UTC).isoformat(),
        "latency_seconds": latency,
        "error_rate": error_rate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parse_args()
    report = run(
        prometheus_url=args.prometheus_url,
        output_path=Path(args.output_path),
        quantile=args.quantile,
        window=args.window,
    )
    print(f"Wrote latency report to {args.output_path}")
    print(f"p{int(args.quantile * 100)} latency: {report['latency_seconds']}")


if __name__ == "__main__":
    main()
