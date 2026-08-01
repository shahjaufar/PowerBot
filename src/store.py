"""SQLite system of record.

Excel is the deliverable, but it is a poor place to *keep* data: it cannot be
written concurrently, it has no unique constraint, and a crash mid-write can
take the workbook with it. So extraction results land here first and the
register is rendered from this table.

Two things fall out of that, both of which matter at a few thousand invoices a
month:
  - Reruns are idempotent. Content-hashing the source file means the same
    document processed twice updates one row instead of creating two.
  - The register can be regenerated at any time — after a template change, a
    mapping fix, or a review pass — without re-running the model.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    file_hash     TEXT PRIMARY KEY,
    source_file   TEXT NOT NULL,
    extracted_at  TEXT NOT NULL,
    payload       TEXT NOT NULL,   -- JSON of the validated field values
    flags         TEXT NOT NULL,   -- JSON array of validation messages
    confidence    REAL NOT NULL,
    reviewed      INTEGER NOT NULL DEFAULT 0,
    exported_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_invoices_confidence ON invoices(confidence);
CREATE INDEX IF NOT EXISTS idx_invoices_exported ON invoices(exported_at);
"""


def file_hash(path: Path) -> str:
    """Content hash of the source document — the deduplication key.

    Content rather than filename, so the same invoice re-sent under a different
    name is still recognised as the one already processed.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        # WAL lets the exporter read while workers are still writing.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def already_processed(self, digest: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM invoices WHERE file_hash = ?", (digest,)
        ).fetchone()
        return row is not None

    def upsert(
        self,
        digest: str,
        source_file: str,
        values: dict[str, Any],
        flags: list[str],
        confidence: float,
    ) -> None:
        payload = json.dumps({k: _encode(v) for k, v in values.items()})
        self.conn.execute(
            """
            INSERT INTO invoices
                (file_hash, source_file, extracted_at, payload, flags, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                source_file  = excluded.source_file,
                extracted_at = excluded.extracted_at,
                payload      = excluded.payload,
                flags        = excluded.flags,
                confidence   = excluded.confidence,
                exported_at  = NULL
            """,
            (
                digest,
                source_file,
                datetime.now().isoformat(timespec="seconds"),
                payload,
                json.dumps(flags),
                confidence,
            ),
        )
        self.conn.commit()

    def pending_export(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM invoices WHERE exported_at IS NULL "
            "ORDER BY extracted_at"
        ).fetchall()

    def all_rows(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM invoices ORDER BY extracted_at"
        ).fetchall()

    def mark_exported(self, digests: Iterable[str]) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        self.conn.executemany(
            "UPDATE invoices SET exported_at = ? WHERE file_hash = ?",
            [(stamp, d) for d in digests],
        )
        self.conn.commit()

    def review_queue(self, threshold: float) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM invoices WHERE confidence < ? AND reviewed = 0 "
            "ORDER BY confidence ASC",
            (threshold,),
        ).fetchall()

    def stats(self) -> dict[str, int]:
        row = self.conn.execute(
            """
            SELECT COUNT(*)                                         AS total,
                   SUM(CASE WHEN flags != '[]' THEN 1 ELSE 0 END)   AS flagged,
                   SUM(CASE WHEN exported_at IS NULL THEN 1 ELSE 0 END) AS unexported
            FROM invoices
            """
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "flagged": row["flagged"] or 0,
            "unexported": row["unexported"] or 0,
        }
