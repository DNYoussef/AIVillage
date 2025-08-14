"""Guardian audit logging with persistent JSON records."""

import datetime
import json
import pathlib
import uuid
from typing import Any

# Create audit directory
_AUDIT_PATH = pathlib.Path("data/guardian_audit")
_AUDIT_PATH.mkdir(parents=True, exist_ok=True)


def log(record: dict[str, Any]) -> None:
    """Log audit record to persistent JSON file.

    Args:
        record: Dictionary containing validation decision and metadata
    """
    # Add unique ID and timestamp if not present
    if "id" not in record:
        record["id"] = str(uuid.uuid4())
    if "timestamp" not in record:
        record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Write to individual JSON file
    filename = f"{record['id']}.json"
    file_path = _AUDIT_PATH / filename

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str, ensure_ascii=False)
    except Exception:
        # Fallback: log to a single append-only file
        fallback_path = _AUDIT_PATH / "audit_log.jsonl"
        with open(fallback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def get_recent_records(hours: int = 24, limit: int = 100) -> list:
    """Get recent audit records.

    Args:
        hours: Hours back to search
        limit: Maximum number of records to return

    Returns:
        List of audit records sorted by timestamp (newest first)
    """
    records = []
    cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)

    # Read from individual JSON files
    for json_file in _AUDIT_PATH.glob("*.json"):
        if json_file.name == "audit_log.jsonl":
            continue

        try:
            with open(json_file, encoding="utf-8") as f:
                record = json.load(f)

            # Check timestamp
            record_time = datetime.datetime.fromisoformat(record.get("timestamp", "1970-01-01T00:00:00"))

            if record_time >= cutoff_time:
                records.append(record)

        except Exception:
            continue  # Skip corrupted files

    # Also read from fallback file if it exists
    fallback_path = _AUDIT_PATH / "audit_log.jsonl"
    if fallback_path.exists():
        try:
            with open(fallback_path, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    record_time = datetime.datetime.fromisoformat(record.get("timestamp", "1970-01-01T00:00:00"))

                    if record_time >= cutoff_time:
                        records.append(record)
        except Exception:
            pass

    # Sort by timestamp (newest first) and limit
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records[:limit]


def get_statistics(hours: int = 24) -> dict[str, Any]:
    """Get audit statistics for the specified time window.

    Args:
        hours: Hours back to analyze

    Returns:
        Dictionary with decision counts and rates
    """
    records = get_recent_records(hours=hours, limit=10000)

    if not records:
        return {"total_validations": 0, "time_window_hours": hours}

    # Count decisions
    decisions = {"APPLY": 0, "QUARANTINE": 0, "REJECT": 0}
    scores = []

    for record in records:
        decision = record.get("decision", "UNKNOWN")
        if decision in decisions:
            decisions[decision] += 1

        score = record.get("score")
        if isinstance(score, int | float):
            scores.append(score)

    total = sum(decisions.values())
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "total_validations": total,
        "time_window_hours": hours,
        "decisions": decisions,
        "decision_rates": {
            "apply_rate": decisions["APPLY"] / total if total > 0 else 0.0,
            "quarantine_rate": decisions["QUARANTINE"] / total if total > 0 else 0.0,
            "reject_rate": decisions["REJECT"] / total if total > 0 else 0.0,
        },
        "average_score": avg_score,
        "score_range": {
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
        },
    }


def cleanup_old_records(days: int = 30):
    """Clean up audit records older than specified days.

    Args:
        days: Records older than this will be deleted
    """
    cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    deleted_count = 0

    # Clean individual JSON files
    for json_file in _AUDIT_PATH.glob("*.json"):
        if json_file.name == "audit_log.jsonl":
            continue

        try:
            with open(json_file, encoding="utf-8") as f:
                record = json.load(f)

            record_time = datetime.datetime.fromisoformat(record.get("timestamp", "1970-01-01T00:00:00"))

            if record_time < cutoff_time:
                json_file.unlink()
                deleted_count += 1

        except Exception:
            # Delete corrupted files too
            json_file.unlink()
            deleted_count += 1

    return deleted_count
