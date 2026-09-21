from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from ..schemas.models import HallucinationPattern, PatternType

logger = logging.getLogger(__name__)

_CREATE_PATTERNS = """
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    description TEXT NOT NULL,
    frequency INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'candidate',
    keywords TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT
);
"""

_CREATE_EXAMPLES = """
CREATE TABLE IF NOT EXISTS pattern_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    added_at TEXT NOT NULL,
    FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
);
"""

_TEMPORAL_KEYWORDS = [
    "invented", "founded", "created", "discovered", "first", "earliest",
    "originated", "began", "started", "established", "launched",
]

_NUMERICAL_KEYWORDS = [
    "percent", "billion", "million", "trillion", "increase", "decrease",
    "growth", "rate", "average", "total", "surpassed", "exceeded",
]

_STATISTICAL_KEYWORDS = [
    "study", "research", "survey", "poll", "findings", "data shows",
    "according to", "evidence suggests", "researchers found",
]

_Causal_KEYWORDS = [
    "caused", "leads to", "results in", "due to", "because",
    "consequence", "effect", "impact", "trigger",
]


class PatternLearner:
    """Rule-based pattern tracker over recurring hallucination categories.

    This is intentionally NOT a trained ML model. It classifies claims with
    keyword/regex heuristics into six categories (temporal, numerical,
    statistical, causal, definition, entity) and tracks historical frequency
    and confidence per domain. A pattern only becomes 'established' once it
    satisfies BOTH gates: frequency >= min_support and
    confidence >= confidence_threshold. Below the gates it remains a
    'candidate'.
    """

    def __init__(
        self,
        db_path: str = "data/patterns.db",
        min_support: int = 3,
        confidence_threshold: float = 0.6,
    ):
        self._db_path = db_path
        self._min_support = min_support
        self._confidence_threshold = confidence_threshold
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(_CREATE_PATTERNS)
        await self._db.execute(_CREATE_EXAMPLES)
        await self._ensure_status_column()
        await self._db.commit()
        logger.info("Pattern learner initialized at %s", self._db_path)

    async def _ensure_status_column(self) -> None:
        """Migrate pre-v1.3 DBs that lack the `status` column."""
        cursor = await self._db.execute("PRAGMA table_info(patterns)")
        columns = [row[1] for row in await cursor.fetchall()]
        if "status" not in columns:
            await self._db.execute(
                "ALTER TABLE patterns ADD COLUMN status TEXT NOT NULL "
                "DEFAULT 'candidate'"
            )

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @staticmethod
    def _classify_claim(claim_text: str) -> list[PatternType]:
        text_lower = claim_text.lower()
        detected: list[PatternType] = []

        temporal_hits = sum(1 for kw in _TEMPORAL_KEYWORDS if kw in text_lower)
        if temporal_hits >= 2:
            detected.append(PatternType.TEMPORAL)

        numerical_hits = sum(
            1 for kw in _NUMERICAL_KEYWORDS if kw in text_lower
        )
        if numerical_hits >= 1 or re.search(r"\d+%", text_lower):
            detected.append(PatternType.NUMERICAL)

        stat_hits = sum(1 for kw in _STATISTICAL_KEYWORDS if kw in text_lower)
        if stat_hits >= 1:
            detected.append(PatternType.STATISTICAL)

        causal_hits = sum(1 for kw in _Causal_KEYWORDS if kw in text_lower)
        if causal_hits >= 1:
            detected.append(PatternType.CAUSAL)

        if re.search(r"\b(is defined as|means that|refers to)\b", text_lower):
            detected.append(PatternType.DEFINITION)

        if re.search(r"\b(is the |was the )\b", text_lower) and not detected:
            detected.append(PatternType.ENTITY)

        if not detected:
            detected.append(PatternType.ENTITY)

        return detected

    def _status_for(self, frequency: int, confidence: float) -> str:
        """A pattern is 'established' only when both gates are met.

        frequency < min_support  -> candidate pattern
        frequency >= min_support and confidence >= confidence_threshold ->
            established pattern
        """
        if (
            frequency >= self._min_support
            and confidence >= self._confidence_threshold
        ):
            return "established"
        return "candidate"

    @staticmethod
    def _extract_keywords(claim_text: str, max_keywords: int = 10) -> list[str]:
        words = re.findall(r"\b[a-z]{3,}\b", claim_text.lower())
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "his", "how",
            "its", "may", "new", "now", "old", "see", "way", "who", "did",
            "get", "let", "say", "she", "too", "use", "that", "with", "have",
            "this", "will", "your", "from", "they", "been", "said", "each",
            "make", "like", "than", "them", "then", "what", "when", "were",
        }
        filtered = [w for w in words if w not in stopwords]
        counts = Counter(filtered)
        return [w for w, _ in counts.most_common(max_keywords)]

    async def observe_claim(
        self,
        claim_text: str,
        domain: str,
        verdict: str,
    ) -> list[HallucinationPattern]:
        if not self._db:
            return []

        is_hallucination = verdict in ("likely_hallucinated", "contradicted")
        pattern_types = self._classify_claim(claim_text)
        keywords = self._extract_keywords(claim_text)
        now = datetime.utcnow().isoformat()
        updated_patterns: list[HallucinationPattern] = []

        for pt in pattern_types:
            cur = await self._db.execute(
                "SELECT pattern_id, frequency, confidence FROM patterns "
                "WHERE pattern_type = ? AND domain = ?",
                (pt.value, domain),
            )
            row = await cur.fetchone()

            if row:
                pid, freq, conf = row
                new_freq = freq + 1
                new_conf = min(
                    1.0,
                    conf + (0.1 if is_hallucination else -0.05),
                )
                new_status = self._status_for(new_freq, new_conf)
                await self._db.execute(
                    "UPDATE patterns SET frequency = ?, confidence = ?, "
                    "status = ?, last_seen_at = ? WHERE pattern_id = ?",
                    (new_freq, max(0.0, new_conf), new_status, now, pid),
                )
                pattern = HallucinationPattern(
                    pattern_id=pid,
                    pattern_type=pt,
                    domain=domain,
                    description=f"Hallucination pattern: {pt.value} in {domain}",
                    frequency=new_freq,
                    confidence=max(0.0, new_conf),
                    status=new_status,
                    keywords=keywords,
                    last_seen_at=datetime.fromisoformat(now),
                )
            else:
                pid = str(uuid.uuid4())
                initial_conf = 0.7 if is_hallucination else 0.3
                new_status = self._status_for(1, initial_conf)
                await self._db.execute(
                    "INSERT INTO patterns "
                    "(pattern_id, pattern_type, domain, description, frequency, "
                    "confidence, status, keywords, created_at, last_seen_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        pid, pt.value, domain,
                        f"Hallucination pattern: {pt.value} in {domain}",
                        1, initial_conf, new_status, json.dumps(keywords), now, now,
                    ),
                )
                pattern = HallucinationPattern(
                    pattern_id=pid,
                    pattern_type=pt,
                    domain=domain,
                    description=f"Hallucination pattern: {pt.value} in {domain}",
                    frequency=1,
                    confidence=initial_conf,
                    status=new_status,
                    keywords=keywords,
                    created_at=datetime.fromisoformat(now),
                    last_seen_at=datetime.fromisoformat(now),
                )

            await self._db.execute(
                "INSERT INTO pattern_examples (pattern_id, claim_text, added_at) "
                "VALUES (?, ?, ?)",
                (pid, claim_text[:2000], now),
            )
            updated_patterns.append(pattern)

        await self._db.commit()
        return updated_patterns

    async def query_patterns(
        self,
        domain: Optional[str] = None,
        pattern_type: Optional[PatternType] = None,
        min_confidence: float = 0.0,
        top_k: int = 20,
    ) -> list[HallucinationPattern]:
        """Return only patterns that satisfy BOTH configured learning gates.

        min_support (frequency) and confidence_threshold are enforced here —
        not just callable min_confidence — so immature candidate patterns are
        never surfaced as established knowledge.
        """
        if not self._db:
            return []

        conditions = ["frequency >= ?", "confidence >= ?"]
        params: list = [
            self._min_support,
            max(min_confidence, self._confidence_threshold),
        ]
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if pattern_type:
            conditions.append("pattern_type = ?")
            params.append(pattern_type.value)

        where = " AND ".join(conditions)
        sql = (
            f"SELECT pattern_id, pattern_type, domain, description, "
            f"frequency, confidence, status, keywords, created_at, last_seen_at "
            f"FROM patterns WHERE {where} "
            f"ORDER BY frequency DESC, confidence DESC LIMIT ?"
        )
        params.append(top_k)

        cur = await self._db.execute(sql, params)
        rows = await cur.fetchall()

        return [
            HallucinationPattern(
                pattern_id=r[0],
                pattern_type=PatternType(r[1]),
                domain=r[2],
                description=r[3],
                frequency=r[4],
                confidence=r[5],
                status=r[6],
                keywords=json.loads(r[7]),
                created_at=datetime.fromisoformat(r[8]),
                last_seen_at=datetime.fromisoformat(r[9]) if r[9] else None,
            )
            for r in rows
        ]

    async def get_pattern(self, pattern_id: str) -> Optional[HallucinationPattern]:
        if not self._db:
            return None
        cur = await self._db.execute(
            "SELECT pattern_id, pattern_type, domain, description, frequency, "
            "confidence, status, keywords, created_at, last_seen_at "
            "FROM patterns WHERE pattern_id = ?",
            (pattern_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return HallucinationPattern(
            pattern_id=row[0],
            pattern_type=PatternType(row[1]),
            domain=row[2],
            description=row[3],
            frequency=row[4],
            confidence=row[5],
            status=row[6],
            keywords=json.loads(row[7]),
            created_at=datetime.fromisoformat(row[8]),
            last_seen_at=datetime.fromisoformat(row[9]) if row[9] else None,
        )

    async def get_examples(self, pattern_id: str, limit: int = 10) -> list[str]:
        if not self._db:
            return []
        cur = await self._db.execute(
            "SELECT claim_text FROM pattern_examples WHERE pattern_id = ? "
            "ORDER BY added_at DESC LIMIT ?",
            (pattern_id, limit),
        )
        rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def get_domain_summary(self, domain: str) -> dict:
        if not self._db:
            return {}
        cur = await self._db.execute(
            "SELECT pattern_type, COUNT(*), AVG(frequency), AVG(confidence) "
            "FROM patterns WHERE domain = ? AND status = 'established' "
            "GROUP BY pattern_type",
            (domain,),
        )
        rows = await cur.fetchall()
        return {
            r[0]: {
                "count": r[1],
                "avg_frequency": round(r[2], 2) if r[2] else 0,
                "avg_confidence": round(r[3], 4) if r[3] else 0,
            }
            for r in rows
        }

    async def get_total_count(self) -> int:
        if not self._db:
            return 0
        cur = await self._db.execute("SELECT COUNT(*) FROM patterns")
        row = await cur.fetchone()
        return row[0]


_pattern_learner_instance: Optional[PatternLearner] = None


def get_pattern_learner(
    db_path: str = "data/patterns.db",
    min_support: int = 3,
    confidence_threshold: float = 0.6,
) -> PatternLearner:
    global _pattern_learner_instance
    if _pattern_learner_instance is None:
        _pattern_learner_instance = PatternLearner(
            db_path=db_path,
            min_support=min_support,
            confidence_threshold=confidence_threshold,
        )
    return _pattern_learner_instance
