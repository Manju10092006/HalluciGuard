from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS storage_operations (
    op_id TEXT PRIMARY KEY,
    op_type TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    subsystem TEXT NOT NULL,
    claim_text TEXT,
    domain TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    reconciled_at TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_storage_ops_status
ON storage_operations(status, subsystem, created_at);
"""

# Subsystems that a single fact touches. The canonical write is recorded in
# SQLite (cache/patterns/trust) and the KG JSON; FAISS is treated as a derived
# index that can always be rebuilt from the journal payload.
DERIVED_INDEX_SUBSYSTEMS = {"vector_store", "cache", "patterns", "trust"}


class StorageJournal:
    """SQLite operation ledger for multi-subsystem (non-transactional) writes.

    A fact write touches Knowledge Graph, FAISS, Pattern DB, Cache DB, and
    Source Trust DB — separate storage backends with no distributed
    transaction covering all of them. Instead of pretending otherwise, this
    journal records the per-subsystem outcome so a partially applied write is
    visible and the derived indexes can be reconciled later.
    """

    def __init__(self, db_path: str = "data/storage_operations.db"):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()
        logger.info("Storage journal initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def record(
        self,
        op_type: str,
        fact_id: str,
        subsystem: str,
        status: str = "pending",
        payload: Optional[dict[str, Any]] = None,
        claim_text: Optional[str] = None,
        domain: Optional[str] = None,
        error: Optional[str] = None,
    ) -> str:
        if not self._db:
            return ""
        op_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            "INSERT INTO storage_operations "
            "(op_id, op_type, fact_id, subsystem, claim_text, domain, payload, "
            "status, error, created_at, reconciled_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                op_id, op_type, fact_id, subsystem,
                claim_text, domain,
                json.dumps(payload or {}, default=str),
                status, error, now,
            ),
        )
        await self._db.commit()
        return op_id

    async def mark_done(self, op_id: str) -> bool:
        return await self._update_status(op_id, "done")

    async def mark_failed(self, op_id: str, error: str) -> bool:
        if not self._db:
            return False
        cur = await self._db.execute(
            "UPDATE storage_operations SET status = 'failed', error = ? "
            "WHERE op_id = ? AND status = 'pending'",
            (error[:500], op_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def mark_reconciled(self, op_id: str) -> bool:
        if not self._db:
            return False
        now = datetime.utcnow().isoformat()
        cur = await self._db.execute(
            "UPDATE storage_operations SET status = 'reconciled', "
            "reconciled_at = ?, error = NULL WHERE op_id = ?",
            (now, op_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def _update_status(self, op_id: str, status: str) -> bool:
        if not self._db:
            return False
        cur = await self._db.execute(
            "UPDATE storage_operations SET status = ? WHERE op_id = ?",
            (status, op_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def get_failed(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._db:
            return []
        cur = await self._db.execute(
            "SELECT op_id, op_type, fact_id, subsystem, claim_text, domain, "
            "payload, status, error, created_at, reconciled_at "
            "FROM storage_operations WHERE status = 'failed' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [self._row_to_map(r) for r in rows]

    async def get_by_fact(self, fact_id: str, limit: int = 20) -> list[dict[str, Any]]:
        if not self._db:
            return []
        cur = await self._db.execute(
            "SELECT op_id, op_type, fact_id, subsystem, claim_text, domain, "
            "payload, status, error, created_at, reconciled_at "
            "FROM storage_operations WHERE fact_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (fact_id, limit),
        )
        rows = await cur.fetchall()
        return [self._row_to_map(r) for r in rows]

    async def get_stats(self) -> dict[str, Any]:
        if not self._db:
            return {}
        totals: dict[str, int] = {}
        by_subsystem: dict[str, dict[str, int]] = {}
        for status in ("done", "failed", "reconciled"):
            cur = await self._db.execute(
                "SELECT subsystem, COUNT(*) FROM storage_operations "
                "WHERE status = ? GROUP BY subsystem",
                (status,),
            )
            for subsystem, count in await cur.fetchall():
                by_subsystem.setdefault(subsystem, {})[status] = count
                totals[status] = totals.get(status, 0) + count
        cur = await self._db.execute("SELECT COUNT(*) FROM storage_operations")
        total = (await cur.fetchone())[0]
        return {
            "total_operations": total,
            "totals": totals,
            "by_subsystem": by_subsystem,
        }

    @staticmethod
    def _row_to_map(r) -> dict[str, Any]:
        payload = r[6]
        try:
            parsed = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        return {
            "op_id": r[0],
            "op_type": r[1],
            "fact_id": r[2],
            "subsystem": r[3],
            "claim_text": r[4],
            "domain": r[5],
            "payload": parsed,
            "status": r[7],
            "error": r[8],
            "created_at": r[9],
            "reconciled_at": r[10],
        }