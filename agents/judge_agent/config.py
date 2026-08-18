"""
HalluciGuard Judge Agent - Central Configuration & Type System
Defines all enums, dataclasses, and configuration for the AI Decision Intelligence Platform.
No hardcoded thresholds. Policy-driven behavioral governance.
"""

import enum
from dataclasses import dataclass, field
from typing import List


class Decision(enum.Enum):
    ACCEPT = "ACCEPT"
    CORRECT = "CORRECT"
    VERIFY_AGAIN = "VERIFY_AGAIN"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"


class ClaimStatus(enum.Enum):
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTED = "CONFLICTED"


class ClaimAction(enum.Enum):
    ACCEPT = "ACCEPT"
    CORRECT = "CORRECT"
    RE_VERIFY = "RE_VERIFY"
    REJECT = "REJECT"


class Severity(enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class EvidenceQuality(enum.Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INSUFFICIENT = "INSUFFICIENT"
    ABSENT = "ABSENT"


class SystemHealth(enum.Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


class SourceTier(enum.Enum):
    OFFICIAL_STANDARD = "OFFICIAL_STANDARD"
    PEER_REVIEWED = "PEER_REVIEWED"
    ENTERPRISE_VENDOR = "ENTERPRISE_VENDOR"
    REPUTABLE_NEWS = "REPUTABLE_NEWS"
    COMMUNITY = "COMMUNITY"
    UNVERIFIED = "UNVERIFIED"


class ConflictType(enum.Enum):
    DIRECT_REFUTATION = "DIRECT_REFUTATION"
    NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"
    ENTITY_MISMATCH = "ENTITY_MISMATCH"
    SAFETY_VIOLATION = "SAFETY_VIOLATION"
    PARTIAL_DISAGREEMENT = "PARTIAL_DISAGREEMENT"
    NO_CONFLICT = "NO_CONFLICT"


@dataclass
class JudgeConfig:
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


DEFAULT_CONFIG = JudgeConfig()
