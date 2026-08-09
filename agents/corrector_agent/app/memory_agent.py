import sqlite3
import json
import time

class MemoryAgent:
    def __init__(self, db_path="corrector_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS hallucination_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    claim_text TEXT,
                    status TEXT,
                    evidence_used TEXT,
                    correction_applied TEXT,
                    domain TEXT
                )
            ''')
            conn.commit()

    def log_hallucination(self, claim_text: str, status: str, evidence_used: str, correction_applied: str, domain: str = "general"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO hallucination_logs (timestamp, claim_text, status, evidence_used, correction_applied, domain)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (int(time.time() * 1000), claim_text, status, evidence_used, correction_applied, domain))
            conn.commit()
