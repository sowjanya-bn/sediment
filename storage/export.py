import json

from .db import get_entry, get_entries_by_date_range, get_all_entries


def export_by_id(entry_id: str) -> dict:
    """Return a single entry dict by ID. Raises KeyError if not found."""
    entry = get_entry(entry_id)
    if entry is None:
        raise KeyError(f"No entry found with id: {entry_id}")
    return entry


def export_by_date_range(start: str, end: str) -> list[dict]:
    """Return entries whose timestamp falls within [start, end] (ISO8601 strings)."""
    # Normalize bare dates to full ISO timestamps for prefix comparison
    if len(start) == 10:
        start = start + "T00:00:00Z"
    if len(end) == 10:
        end = end + "T23:59:59Z"
    return get_entries_by_date_range(start, end)


def export_all() -> list[dict]:
    """Return all entries ordered by timestamp."""
    return get_all_entries()


def to_ndjson(entries: list[dict]) -> str:
    """Serialize a list of entry dicts to newline-delimited JSON."""
    lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
    return "\n".join(lines)
