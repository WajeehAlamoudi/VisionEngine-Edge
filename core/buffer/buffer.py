from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import aiosqlite

from core.config import BufferConfig
from .types import BufferedRow


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS buffer (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name  TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    created_at  REAL    NOT NULL,
    sent_at     REAL
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_status_created
ON buffer (status, created_at)
"""


class Buffer:
    """
    SQLite-backed local buffer.

    Rows flow through three states:
      pending → sent after a successful API push
      pending → pending again after a failed push (retried next flush)
      sent rows are deleted after delete_after_hours

    When the DB exceeds max_size_mb, oldest sent rows are deleted first.
    If still over the limit, oldest pending rows are dropped (data loss warning).
    """

    def __init__(self, cfg: BufferConfig) -> None:
        self._path = Path(cfg.path)
        self._max_bytes = cfg.max_size_mb * 1024 * 1024
        self._delete_after = cfg.delete_after_hours * 3600
        self._db: aiosqlite.Connection | None = None

    async def start(self) -> None:
        """Open (or create) the SQLite database and set up the schema."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()

    async def stop(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def write(self, rows: list[dict]) -> None:
        """
        Persist detection rows.

        Each item must have:
          "table" → target table name in the branch schema
          "row"   → dict with detection fields
        """
        if not rows:
            return
        now = time.time()
        await self._db.executemany(
            "INSERT INTO buffer (table_name, payload, status, created_at) "
            "VALUES (?, ?, 'pending', ?)",
            [(r["table"], json.dumps(r["row"]), now) for r in rows],
        )
        await self._db.commit()
        await self._enforce_size_limit()

    async def get_pending(self, limit: int = 500) -> list[BufferedRow]:
        """Return up to `limit` pending rows, oldest first."""
        async with self._db.execute(
            "SELECT id, table_name, payload, created_at FROM buffer "
            "WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ) as cursor:
            return [
                BufferedRow(
                    id=row["id"],
                    table=row["table_name"],
                    row=json.loads(row["payload"]),
                    created_at=row["created_at"],
                )
                for row in await cursor.fetchall()
            ]

    async def mark_sent(self, ids: list[int]) -> None:
        """Mark rows as successfully pushed."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        await self._db.execute(
            f"UPDATE buffer SET status='sent', sent_at=? WHERE id IN ({placeholders})",
            [time.time(), *ids],
        )
        await self._db.commit()

    async def mark_failed(self, ids: list[int]) -> None:
        """Reset failed rows back to pending so they are retried on the next flush."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        await self._db.execute(
            f"UPDATE buffer SET status='pending' WHERE id IN ({placeholders})",
            ids,
        )
        await self._db.commit()

    async def purge_old_sent(self) -> int:
        """Delete sent rows older than delete_after_hours. Returns count deleted."""
        cutoff = time.time() - self._delete_after
        async with self._db.execute(
            "DELETE FROM buffer WHERE status='sent' AND sent_at < ?",
            (cutoff,),
        ) as cursor:
            count = cursor.rowcount
        await self._db.commit()
        return count

    async def stats(self) -> dict:
        """Return counts and DB size for health reporting."""
        async with self._db.execute(
            "SELECT status, COUNT(*) AS cnt FROM buffer GROUP BY status"
        ) as cursor:
            counts = {row["status"]: row["cnt"] for row in await cursor.fetchall()}

        size_mb = (
            round(self._path.stat().st_size / (1024 * 1024), 2)
            if self._path.exists()
            else 0.0
        )
        return {
            "pending": counts.get("pending", 0),
            "sent":    counts.get("sent", 0),
            "size_mb": size_mb,
        }

    async def _enforce_size_limit(self) -> None:
        if self._max_bytes <= 0 or not self._path.exists():
            return
        if self._path.stat().st_size <= self._max_bytes:
            return

        await self._db.execute(
            "DELETE FROM buffer WHERE id IN ("
            "  SELECT id FROM buffer WHERE status='sent' "
            "  ORDER BY created_at ASC LIMIT 1000"
            ")"
        )
        await self._db.commit()
        await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        if self._path.stat().st_size <= self._max_bytes:
            return

        print(
            "WARNING  buffer: size limit exceeded after purging sent rows — "
            "dropping oldest pending rows. Increase max_size_mb or fix connectivity.",
            file=sys.stderr,
        )
        await self._db.execute(
            "DELETE FROM buffer WHERE id IN ("
            "  SELECT id FROM buffer WHERE status='pending' "
            "  ORDER BY created_at ASC LIMIT 1000"
            ")"
        )
        await self._db.commit()
        await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
