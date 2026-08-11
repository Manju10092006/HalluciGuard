from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiosqlite

from ..schemas.models import CacheStats, CachedVerification

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS verification_cache (
    cache_key TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    verdict TEXT NOT NULL,
    evidence_summary TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_count INTEGER NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""

_CREATE_HITS_TABLE = """
CREATE TABLE IF NOT EXISTS cache_hits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT NOT NULL,
    hit_at TEXT NOT NULL,
    FOREIGN KEY (cache_key) REFERENCES verification_cache(cache_key)
);
"""


class VerificationCache:
    """Async SQLite cache for verification results with TTL expiration."""

    def __init__(self, db_path: str = "data/verification_cache.db", ttl: int = 86400):
        self._db_path = db_path
        self._ttl = ttl
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_HITS_TABLE)
        await self._db.commit()
        logger.info("Verification cache initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @staticmethod
    def _make_key(domain: str, claim_text: str) -> str:
        normalized = " ".join(claim_text.lower().split())
        raw = f"{domain}:{normalized}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get(self, domain: str, claim_text: str) -> Optional[CachedVerification]:
        if not self._db:
            return None

        key = self._make_key(domain, claim_text)
        cursor = await self._db.execute(
            "SELECT * FROM verification_cache WHERE cache_key = ?", (key,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        expires_at = datetime.fromisoformat(row[9])
        if datetime.utcnow() > expires_at:
            await self._db.execute(
                "DELETE FROM verification_cache WHERE cache_key = ?", (key,)
            )
            await self._db.commit()
            return None

        await self._db.execute(
            "UPDATE verification_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (key,),
        )
        await self._db.execute(
            "INSERT INTO cache_hits (cache_key, hit_at) VALUES (?, ?)",
            (key, datetime.utcnow().isoformat()),
        )
        await self._db.commit()

        return CachedVerification(
            cache_key=row[0],
            domain=row[1],
            claim_text=row[2],
            verdict=row[3],
            evidence_summary=row[4],
            confidence=row[5],
            source_count=row[6],
            created_at=datetime.fromisoformat(row[8]),
            expires_at=expires_at,
        )

    async def set(
        self,
        domain: str,
        claim_text: str,
        verdict: str,
        evidence_summary: str,
        confidence: float,
        source_count: int,
    ) -> str:
        if not self._db:
            raise RuntimeError("Cache not initialized")

        key = self._make_key(domain, claim_text)
        now = datetime.utcnow()
        expires = now + timedelta(seconds=self._ttl)

        await self._db.execute(
            """
            INSERT INTO verification_cache
                (cache_key, domain, claim_text, verdict, evidence_summary,
                 confidence, source_count, hit_count, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                verdict = excluded.verdict,
                evidence_summary = excluded.evidence_summary,
                confidence = excluded.confidence,
                source_count = excluded.source_count,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
            """,
            (key, domain, claim_text, verdict, evidence_summary,
             confidence, source_count, now.isoformat(), expires.isoformat()),
        )
        await self._db.commit()
        return key

    async def invalidate(self, domain: str, claim_text: str) -> bool:
        if not self._db:
            return False
        key = self._make_key(domain, claim_text)
        cursor = await self._db.execute(
            "DELETE FROM verification_cache WHERE cache_key = ?", (key,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def invalidate_domain(self, domain: str) -> int:
        if not self._db:
            return 0
        cursor = await self._db.execute(
            "DELETE FROM verification_cache WHERE domain = ?", (domain,)
        )
        await self._db.commit()
        return cursor.rowcount

    async def get_stats(self) -> CacheStats:
        if not self._db:
            return CacheStats(
                total_entries=0, total_hits=0, total_misses=0,
                hit_rate=0.0, oldest_entry=None, newest_entry=None,
                domain_breakdown={},
            )

        cur = await self._db.execute("SELECT COUNT(*) FROM verification_cache")
        total_entries = (await cur.fetchone())[0]

        cur = await self._db.execute("SELECT COUNT(*) FROM cache_hits")
        total_hits = (await cur.fetchone())[0]

        cur = await self._db.execute("SELECT MIN(created_at), MAX(created_at) FROM verification_cache")
        row = await cur.fetchone()
        oldest = datetime.fromisoformat(row[0]) if row[0] else None
        newest = datetime.fromisoformat(row[1]) if row[1] else None

        cur = await self._db.execute(
            "SELECT domain, COUNT(*) FROM verification_cache GROUP BY domain"
        )
        domain_rows = await cur.fetchall()
        domain_breakdown = {r[0]: r[1] for r in domain_rows}

        total_requests = total_entries + total_hits
        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

        return CacheStats(
            total_entries=total_entries,
            total_hits=total_hits,
            total_misses=total_entries,
            hit_rate=round(hit_rate, 4),
            oldest_entry=oldest,
            newest_entry=newest,
            domain_breakdown=domain_breakdown,
        )

    async def cleanup_expired(self) -> int:
        if not self._db:
            return 0
        now = datetime.utcnow().isoformat()
        cursor = await self._db.execute(
            "DELETE FROM verification_cache WHERE expires_at < ?", (now,)
        )
        await self._db.commit()
        return cursor.rowcount


_verification_cache_instance: Optional[VerificationCache] = None


def get_verification_cache(
    db_path: str = "data/verification_cache.db", ttl: int = 86400
) -> VerificationCache:
    global _verification_cache_instance
    if _verification_cache_instance is None:
        _verification_cache_instance = VerificationCache(db_path=db_path, ttl=ttl)
    return _verification_cache_instance
