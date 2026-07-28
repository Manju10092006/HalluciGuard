from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from ..schemas.models import SourceTrustRecord, TrustChangeReason, TrustUpdate

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS source_trust (
    source_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    trust_score REAL NOT NULL DEFAULT 0.5,
    total_verifications INTEGER DEFAULT 0,
    correct_count INTEGER DEFAULT 0,
    incorrect_count INTEGER DEFAULT 0,
    conflict_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_CREATE_HISTORY = """
CREATE TABLE IF NOT EXISTS trust_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    old_score REAL NOT NULL,
    new_score REAL NOT NULL,
    reason TEXT NOT NULL,
    delta REAL NOT NULL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES source_trust(source_id)
);
"""

_TRUST_DELTAS = {
    TrustChangeReason.VERIFIED_CORRECT: 0.05,
    TrustChangeReason.VERIFIED_INCORRECT: -0.15,
    TrustChangeReason.CONFLICTING_EVIDENCE: -0.05,
    TrustChangeReason.INSUFFICIENT_EVIDENCE: -0.02,
    TrustChangeReason.EXPLICIT_FEEDBACK: 0.0,
}


class SourceTrustManager:
    """Evolves source reliability scores based on verification outcomes."""

    def __init__(
        self,
        db_path: str = "data/source_trust.db",
        prior: float = 0.5,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
    ):
        self._db_path = db_path
        self._prior = prior
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_HISTORY)
        await self._db.commit()
        logger.info("Source trust manager initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def get_trust(self, source_id: str) -> Optional[SourceTrustRecord]:
        if not self._db:
            return None
        cur = await self._db.execute(
            "SELECT * FROM source_trust WHERE source_id = ?", (source_id,)
        )
        row = await cur.fetchone()
        if not row:
            return None
        return SourceTrustRecord(
            source_id=row[0],
            source_name=row[1],
            domain=row[2],
            trust_score=row[3],
            total_verifications=row[4],
            correct_count=row[5],
            incorrect_count=row[6],
            conflict_count=row[7],
            created_at=datetime.fromisoformat(row[8]),
            updated_at=datetime.fromisoformat(row[9]),
        )

    async def update_trust(
        self,
        source_id: str,
        source_name: str,
        domain: str,
        reason: TrustChangeReason,
        evidence_count: int = 1,
    ) -> TrustUpdate:
        if not self._db:
            raise RuntimeError("Source trust manager not initialized")

        now = datetime.utcnow().isoformat()
        existing = await self.get_trust(source_id)

        if existing:
            old_score = existing.trust_score
            base_delta = _TRUST_DELTAS.get(reason, 0.0)
            scaled_delta = base_delta * self._learning_rate * min(evidence_count, 5)
            new_score = max(0.0, min(1.0, old_score + scaled_delta))

            counters = {
                TrustChangeReason.VERIFIED_CORRECT: "correct_count",
                TrustChangeReason.VERIFIED_INCORRECT: "incorrect_count",
                TrustChangeReason.CONFLICTING_EVIDENCE: "conflict_count",
            }
            counter_col = counters.get(reason)
            if counter_col:
                await self._db.execute(
                    f"UPDATE source_trust SET {counter_col} = {counter_col} + 1, "
                    "total_verifications = total_verifications + 1, "
                    "trust_score = ?, updated_at = ? WHERE source_id = ?",
                    (new_score, now, source_id),
                )
            else:
                await self._db.execute(
                    "UPDATE source_trust SET trust_score = ?, updated_at = ? "
                    "WHERE source_id = ?",
                    (new_score, now, source_id),
                )
        else:
            base_delta = _TRUST_DELTAS.get(reason, 0.0)
            scaled_delta = base_delta * self._learning_rate
            new_score = max(0.0, min(1.0, self._prior + scaled_delta))

            await self._db.execute(
                "INSERT INTO source_trust "
                "(source_id, source_name, domain, trust_score, total_verifications, "
                "correct_count, incorrect_count, conflict_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                (
                    source_id, source_name, domain, new_score,
                    1 if reason == TrustChangeReason.VERIFIED_CORRECT else 0,
                    1 if reason == TrustChangeReason.VERIFIED_INCORRECT else 0,
                    1 if reason == TrustChangeReason.CONFLICTING_EVIDENCE else 0,
                    now, now,
                ),
            )

        await self._db.execute(
            "INSERT INTO trust_history "
            "(source_id, old_score, new_score, reason, delta, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                source_id,
                existing.trust_score if existing else self._prior,
                new_score,
                reason.value,
                new_score - (existing.trust_score if existing else self._prior),
                now,
            ),
        )
        await self._db.commit()

        return TrustUpdate(
            source_id=source_id,
            old_score=existing.trust_score if existing else self._prior,
            new_score=new_score,
            reason=reason,
            delta=new_score - (existing.trust_score if existing else self._prior),
        )

    async def get_domain_sources(
        self, domain: str, min_trust: float = 0.0
    ) -> list[SourceTrustRecord]:
        if not self._db:
            return []
        cur = await self._db.execute(
            "SELECT * FROM source_trust WHERE domain = ? AND trust_score >= ? "
            "ORDER BY trust_score DESC",
            (domain, min_trust),
        )
        rows = await cur.fetchall()
        return [
            SourceTrustRecord(
                source_id=r[0], source_name=r[1], domain=r[2],
                trust_score=r[3], total_verifications=r[4],
                correct_count=r[5], incorrect_count=r[6],
                conflict_count=r[7],
                created_at=datetime.fromisoformat(r[8]),
                updated_at=datetime.fromisoformat(r[9]),
            )
            for r in rows
        ]

    async def get_top_sources(
        self, domain: Optional[str] = None, limit: int = 20
    ) -> list[SourceTrustRecord]:
        if not self._db:
            return []
        if domain:
            cur = await self._db.execute(
                "SELECT * FROM source_trust WHERE domain = ? "
                "ORDER BY trust_score DESC LIMIT ?",
                (domain, limit),
            )
        else:
            cur = await self._db.execute(
                "SELECT * FROM source_trust ORDER BY trust_score DESC LIMIT ?",
                (limit,),
            )
        rows = await cur.fetchall()
        return [
            SourceTrustRecord(
                source_id=r[0], source_name=r[1], domain=r[2],
                trust_score=r[3], total_verifications=r[4],
                correct_count=r[5], incorrect_count=r[6],
                conflict_count=r[7],
                created_at=datetime.fromisoformat(r[8]),
                updated_at=datetime.fromisoformat(r[9]),
            )
            for r in rows
        ]

    async def get_trust_history(
        self, source_id: str, limit: int = 50
    ) -> list[dict]:
        if not self._db:
            return []
        cur = await self._db.execute(
            "SELECT old_score, new_score, reason, delta, recorded_at "
            "FROM trust_history WHERE source_id = ? "
            "ORDER BY recorded_at DESC LIMIT ?",
            (source_id, limit),
        )
        rows = await cur.fetchall()
        return [
            {
                "old_score": r[0],
                "new_score": r[1],
                "reason": r[2],
                "delta": r[3],
                "recorded_at": r[4],
            }
            for r in rows
        ]

    async def apply_global_decay(self) -> int:
        if not self._db:
            return 0
        cur = await self._db.execute(
            "UPDATE source_trust SET trust_score = MAX(0.1, trust_score * ?) "
            "WHERE trust_score > 0.1",
            (1.0 - self._decay_rate,),
        )
        await self._db.commit()
        return cur.rowcount

    async def get_total_sources(self) -> int:
        if not self._db:
            return 0
        cur = await self._db.execute("SELECT COUNT(*) FROM source_trust")
        row = await cur.fetchone()
        return row[0]


_source_trust_instance: Optional[SourceTrustManager] = None


def get_source_trust_manager(
    db_path: str = "data/source_trust.db",
    prior: float = 0.5,
    learning_rate: float = 0.1,
    decay_rate: float = 0.01,
) -> SourceTrustManager:
    global _source_trust_instance
    if _source_trust_instance is None:
        _source_trust_instance = SourceTrustManager(
            db_path=db_path,
            prior=prior,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
        )
    return _source_trust_instance
