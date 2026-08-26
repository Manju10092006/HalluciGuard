"""
HalluciGuard Judge Agent - Central Configuration & Type System
Defines configuration and re-exports canonical Phase-1 contracts.
"""

import enum
from dataclasses import dataclass, field
from typing import List, Dict, Any

from orchestration.schemas import (
    JudgeResult,
    CorrectionRequest,
    ReverificationResult,
    VerifierResult,
    ClaimReport,
    Evidence,
    DetectorResult,
    JudgeDecision,
    SeverityLevel,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
)


class Decision(enum.Enum):
    """Judge decision outcomes for a verification result."""
    ACCEPT = "ACCEPT"
    CORRECT = "CORRECT"
    VERIFY_AGAIN = "VERIFY_AGAIN"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"


class ClaimStatus(enum.Enum):
    """Status classification for individual claims."""
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTED = "CONFLICTED"


class ClaimAction(enum.Enum):
    """Actions that can be taken on a claim."""
    ACCEPT = "ACCEPT"
    CORRECT = "CORRECT"
    RE_VERIFY = "RE_VERIFY"
    REJECT = "REJECT"


class Severity(enum.Enum):
    """Severity levels for risk assessment."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class EvidenceQuality(enum.Enum):
    """Quality assessment levels for evidence."""
    AUTHORITATIVE = "AUTHORITATIVE"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INSUFFICIENT = "INSUFFICIENT"
    ABSENT = "ABSENT"


class SystemHealth(enum.Enum):
    """Overall system health status."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


class SourceTier(enum.Enum):
    """Reliability tier classification for information sources."""
    OFFICIAL_STANDARD = "OFFICIAL_STANDARD"
    PEER_REVIEWED = "PEER_REVIEWED"
    ENTERPRISE_VENDOR = "ENTERPRISE_VENDOR"
    REPUTABLE_NEWS = "REPUTABLE_NEWS"
    COMMUNITY = "COMMUNITY"
    UNVERIFIED = "UNVERIFIED"


class ConflictType(enum.Enum):
    """Types of conflicts that can be detected in evidence."""
    DIRECT_REFUTATION = "DIRECT_REFUTATION"
    NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"
    ENTITY_MISMATCH = "ENTITY_MISMATCH"
    SAFETY_VIOLATION = "SAFETY_VIOLATION"
    PARTIAL_DISAGREEMENT = "PARTIAL_DISAGREEMENT"
    NO_CONFLICT = "NO_CONFLICT"


@dataclass
class JudgeConfig:
    """Configuration settings for the Judge Agent."""
    supported_domains: List[str] = field(default_factory=lambda: [
        "Healthcare", "Cybersecurity", "Finance", "Law",
        "Scientific Research", "General Knowledge", "Entertainment"
    ])
    enable_audit: bool = True
    enable_observability: bool = True
    max_verification_retries: int = 2
    log_level: str = "INFO"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    use_huggingface: bool = True
    circuit_breaker_error_threshold: int = 3

    @property
    def default_nli_model(self) -> str:
        """Return the configured NLI model name."""
        return self.nli_model


DEFAULT_CONFIG = JudgeConfig()
