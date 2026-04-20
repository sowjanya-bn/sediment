import json
import os
import sqlite3
from pathlib import Path

_SEDIMENT_DIR = Path(os.path.expanduser("~/.sediment"))
_DB_PATH = _SEDIMENT_DIR / "sediment.db"

_CREATE_CORRECTIONS_SQL = """
CREATE TABLE IF NOT EXISTS corrections (
  cr_id TEXT PRIMARY KEY,
  entry_id TEXT NOT NULL,
  entry_hash TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  cr_type TEXT NOT NULL,
  field_target TEXT NOT NULL,
  previous_value TEXT NOT NULL,
  corrected_value TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  note TEXT
);
CREATE INDEX IF NOT EXISTS idx_corrections_entry ON corrections(entry_id);
"""


def _connect() -> sqlite3.Connection:
    _SEDIMENT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_corrections_table() -> None:
    conn = _connect()
    with conn:
        conn.executescript(_CREATE_CORRECTIONS_SQL)
    conn.close()


def save_correction(cr: dict) -> None:
    _init_corrections_table()
    conn = _connect()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO corrections
              (cr_id, entry_id, entry_hash, schema_version, cr_type,
               field_target, previous_value, corrected_value, timestamp, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cr["cr_id"],
                cr["entry_id"],
                cr["entry_hash"],
                cr["schema_version"],
                cr["cr_type"],
                cr["field_target"],
                cr["previous_value"],
                cr["corrected_value"],
                cr["timestamp"],
                cr.get("note"),
            ),
        )
    conn.close()


def get_corrections_for_entry(entry_id: str) -> list[dict]:
    _init_corrections_table()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM corrections WHERE entry_id = ? ORDER BY timestamp",
        (entry_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_corrections() -> list[dict]:
    _init_corrections_table()
    conn = _connect()
    rows = conn.execute("SELECT * FROM corrections ORDER BY timestamp").fetchall()
    conn.close()
    return [dict(r) for r in rows]
