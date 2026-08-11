import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

@dataclass
class VerificationLogRecord:
    query_id: str
    request_id: str
    domain: str
    stages_timing: dict[str, int]
    adapters_called: list[str]
    adapters_failed: list[str]
    verdict: str
    confidence: float
    total_latency_ms: int
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "timestamp": self.formatTime(record, self.datefmt)
        }
        if hasattr(record, "verification_data"):
            log_data["verification_data"] = asdict(getattr(record, "verification_data"))
            
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    return logger

def log_verification_request(logger: logging.Logger, record: VerificationLogRecord) -> None:
    logger.info(
        f"Verification request completed: {record.query_id}",
        extra={"verification_data": record}
    )
