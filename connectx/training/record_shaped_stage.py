from __future__ import annotations

import argparse
import json
from pathlib import Path

from connectx.training.run_manifest import append_training_journal


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a shaped-training stage record to the journal.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", required=True, choices=["started", "completed", "failed"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--journal", default="results/shaped/training_journal.csv")
    parser.add_argument("--details-json", default="{}")
    args = parser.parse_args()
    details = json.loads(args.details_json)
    append_training_journal(
        args.journal,
        stage=args.stage,
        status=args.status,
        run_dir=args.run_dir,
        details=details,
    )
    print(f"Recorded {args.stage} -> {args.status}")


if __name__ == "__main__":
    main()
