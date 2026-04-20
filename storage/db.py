import json
import os
import sqlite3
from pathlib import Path

_SEDIMENT_DIR = Path(os.path.expanduser("~/.sediment"))
_DB_PATH = _SEDIMENT_DIR / "sediment.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  schema_version TEXT NOT NULL DEFAULT '1.1',
  raw_text TEXT NOT NULL,
  input_type TEXT NOT NULL DEFAULT 'affective',
  valence REAL,
  arousal TEXT,
  emotion_label TEXT,
  themes TEXT,
  intensity TEXT,
  salient_focus TEXT,
  state_direction TEXT,
  low_confidence INTEGER DEFAULT 0,
  confidence_score REAL,
  entry_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON entries(timestamp);
"""

_MIGRATE_SQL = [
    "ALTER TABLE entries ADD COLUMN input_type TEXT NOT NULL DEFAULT 'affective'",
]


def _connect() -> sqlite3.Connection:
    _SEDIMENT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    with conn:
        conn.executescript(_CREATE_TABLE_SQL)
        # Migrate existing DBs that predate v1.1 columns
        for stmt in _MIGRATE_SQL:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
    conn.close()


def save_entry(record: dict) -> None:
    init_db()
    conn = _connect()
    themes_json = json.dumps(record["themes"]) if isinstance(record["themes"], list) else record["themes"]
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO entries
              (id, timestamp, schema_version, raw_text, input_type, valence, arousal,
               emotion_label, themes, intensity, salient_focus, state_direction,
               low_confidence, confidence_score, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["timestamp"],
                record["schema_version"],
                record["raw_text"],
                record.get("input_type", "affective"),
                record["valence"],
                record["arousal"],
                record["emotion_label"],
                themes_json,
                record["intensity"],
                record["salient_focus"],
                record["state_direction"],
                int(record["low_confidence"]),
                record["confidence_score"],
                record["entry_hash"],
            ),
        )
    conn.close()


def get_entry(entry_id: str) -> dict | None:
    init_db()
    conn = _connect()
    row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_dict(row)


def get_entries_by_date_range(start: str, end: str) -> list[dict]:
    init_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM entries WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
        (start, end),
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_all_entries() -> list[dict]:
    init_db()
    conn = _connect()
    rows = conn.execute("SELECT * FROM entries ORDER BY timestamp").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if isinstance(d.get("themes"), str):
        try:
            d["themes"] = json.loads(d["themes"])
        except (json.JSONDecodeError, TypeError):
            d["themes"] = [d["themes"]]
    d["low_confidence"] = bool(d.get("low_confidence", 0))
    return d
