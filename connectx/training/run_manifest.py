from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_manifest(run_dir: str | Path, **fields: Any) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    payload: dict[str, Any] = {}
    if path.exists():
        payload.update(json.loads(path.read_text()))
    payload.update(fields)
    if "created_at" not in payload:
        payload["created_at"] = _utc_now()
    payload["updated_at"] = _utc_now()
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def reward_shaping_fields(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"reward_shaping": False}
    from connectx.agents.reward_shaping import RewardShapingConfig

    return {"reward_shaping": True, "reward_shaping_config": asdict(RewardShapingConfig())}


def append_training_journal(
    journal_path: str | Path,
    *,
    stage: str,
    status: str,
    run_dir: str | Path,
    details: dict[str, Any] | None = None,
) -> None:
    import csv

    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _utc_now(),
        "stage": stage,
        "status": status,
        "run_dir": str(run_dir),
        **(details or {}),
    }
    exists = journal_path.exists()
    with journal_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
