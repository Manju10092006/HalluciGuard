from __future__ import annotations
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging

class SqliteCache:
    """SQLite-based cache for query results."""

    def __init__(self, db_path: str = "verification_cache.db", ttl_seconds: int = 86400) -> None:
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds

        
    async def init_db(self) -> None:
        """Initialize the database table if it doesn't exist."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS verification_cache (
                        cache_key TEXT PRIMARY KEY,
                        payload TEXT,
                        timestamp TEXT,
                        domain TEXT
                    )
                ''')
                await db.commit()
        except ImportError:
            logging.warning("aiosqlite not installed. SqliteCache will fail.")
        except Exception as e:
            logging.error(f"Error initializing cache DB: {e}")

    def _normalize_key(self, domain: str, query: str) -> str:
        """Generate a deterministic cache key preserving query semantics."""
        normalized = query.lower().strip()
        key_input = f"{domain}:{normalized}"
        return hashlib.sha256(key_input.encode('utf-8')).hexdigest()

    async def get(self, domain: str, query: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result if it hasn't expired."""
        key = self._normalize_key(domain, query)
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('SELECT payload, timestamp FROM verification_cache WHERE cache_key = ?', (key,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if not row:
                        return None
                        
                    payload_json, timestamp_str = row
                    
                    # Check TTL
                    cached_time = datetime.fromisoformat(timestamp_str)
                    now = datetime.now(timezone.utc)
                    if (now - cached_time).total_seconds() > self.ttl_seconds:
                        # Expired
                        await self.invalidate(domain, query)
                        return None
                        
                    return json.loads(payload_json)
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            return None

    async def set(self, domain: str, query: str, payload: Dict[str, Any]) -> None:
        """Store a result in the cache."""
        key = self._normalize_key(domain, query)
        timestamp = datetime.now(timezone.utc).isoformat()
        payload_str = json.dumps(payload)
        
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO verification_cache (cache_key, payload, timestamp, domain)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        payload=excluded.payload,
                        timestamp=excluded.timestamp,
                        domain=excluded.domain
                ''', (key, payload_str, timestamp, domain))
                await db.commit()
        except Exception as e:
            logging.error(f"Cache set error: {e}")

    async def invalidate(self, domain: str, query: str) -> None:
        """Remove a specific entry from the cache."""
        key = self._normalize_key(domain, query)
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('DELETE FROM verification_cache WHERE cache_key = ?', (key,))
                await db.commit()
        except Exception as e:
            logging.error(f"Cache invalidate error: {e}")

    async def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache."""
        try:
            import aiosqlite
            count = 0
            async with aiosqlite.connect(self.db_path) as db:
                # Iterate and check TTL
                async with db.execute('SELECT cache_key, timestamp FROM verification_cache') as cursor:
                    rows = await cursor.fetchall()
                    
                now = datetime.now(timezone.utc)
                to_delete = []
                for row in rows:
                    key, timestamp_str = row
                    cached_time = datetime.fromisoformat(timestamp_str)
                    if (now - cached_time).total_seconds() > self.ttl_seconds:
                        to_delete.append((key,))
                
                if to_delete:
                    await db.executemany('DELETE FROM verification_cache WHERE cache_key = ?', to_delete)
                    await db.commit()
                    count = len(to_delete)
                    
            return count
        except Exception as e:
            logging.error(f"Cache cleanup error: {e}")
            return 0

    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM verification_cache') as cursor:
                    row = await cursor.fetchone()
                    count, oldest, newest = row if row else (0, None, None)
                    
            return {
                'total_entries': count,
                'oldest_entry': oldest,
                'newest_entry': newest,
                # Hit/miss counts are usually maintained by a MetricsCollector, not the DB itself,
                # so we just return DB stats here.
            }
        except Exception as e:
            logging.error(f"Cache stats error: {e}")
            return {'total_entries': 0}
