"""Persist analysis runs as JSON history records."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_DIR = Path(__file__).resolve().parent / "history"
HISTORY_INDEX = HISTORY_DIR / "history.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_run(record: dict[str, Any]) -> Path:
    """Save a single run as its own JSON file and append to the rolling index.

    Returns the path of the per-run JSON file.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = dict(record)  # shallow copy
    record.setdefault("timestamp_utc", _utcnow_iso())
    video_name = record.get("video", {}).get("filename", "video")
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in video_name)
    run_path = HISTORY_DIR / f"run_{ts}_{safe_name}.json"
    run_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    index = []
    if HISTORY_INDEX.exists():
        try:
            index = json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))
            if not isinstance(index, list):
                index = []
        except json.JSONDecodeError:
            index = []
    index.append({
        "timestamp_utc": record["timestamp_utc"],
        "run_file": run_path.name,
        "video": record.get("video", {}),
        "provider": record.get("provider", {}),
        "surgery_type": record.get("analysis", {}).get("surgery_type", {}),
        "summary": record.get("analysis", {}).get("video_summary", {}),
    })
    HISTORY_INDEX.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_path


def load_index() -> list[dict[str, Any]]:
    if not HISTORY_INDEX.exists():
        return []
    try:
        data = json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []
