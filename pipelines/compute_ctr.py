import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.repositories import UserEventRepository
from app.db.session import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute click-through rate from user events.")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--output-path", required=True, help="Where to write the CTR JSON")
    return parser.parse_args()


def compute_ctr(db: Session, window_hours: int) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    repo = UserEventRepository(db)

    impressions = repo.count_events_since("impression", since)
    clicks = repo.count_events_since("click", since)
    ctr = clicks / impressions if impressions > 0 else None

    return {
        "window_hours": window_hours,
        "since": since.isoformat(),
        "computed_at": datetime.now(UTC).isoformat(),
        "impressions": impressions,
        "clicks": clicks,
        "ctr": ctr,
    }


def run(output_path: Path, window_hours: int) -> dict[str, Any]:
    db = SessionLocal()
    try:
        report = compute_ctr(db, window_hours)
    finally:
        db.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parse_args()
    report = run(Path(args.output_path), args.window_hours)
    print(f"Wrote CTR report to {args.output_path}")
    print(f"CTR: {report['ctr']} ({report['clicks']}/{report['impressions']})")


if __name__ == "__main__":
    main()
