"""Pydantic schemas for Verifier Agent input/output contracts."""
from .models import (
    Passage, SuspiciousClaim, VerifierInputV2, EvidenceItem,
    ClaimReport, VerifierOutputV2, AdapterMetadata, PipelineStageStatus,
    AdapterHealthStatus, DomainStatistics
)
