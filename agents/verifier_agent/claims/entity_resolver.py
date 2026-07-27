"""
HalluciGuard Entity Resolver — Entity-Aware Query Resolution & Canonicalization.

Extracts and canonicalizes domain-specific named entities from claim text prior to retrieval:
  - Cybersecurity: CVE IDs (e.g. CVE-2021-44228), ATT&CK technique IDs (e.g. T1059), threat/malware names.
  - Finance: Company names, stock tickers (AAPL, TSLA, MSFT, AMZN, GOOGL), SEC CIK numbers.
  - Healthcare / Pharmacy / Medicine: Drug names, medical conditions, active ingredients.
  - Legal: Statutes, legal acts (IPC, CrPC, GDPR, US Code), section/article numbers.
  - AI Research / Computer Science: Model names (Transformer, LoRA, DeBERTa, BERT, GPT-4), core concepts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class EntityResolution:
    """Canonical representation of resolved entities extracted from a claim."""

    original_claim: str
    domain: str
    primary_entity: Optional[str] = None
    entity_type: Optional[str] = None  # e.g., "CVE", "TICKER", "DRUG", "STATUTE"
    identifiers: Dict[str, str] = field(default_factory=dict)  # e.g., {"cve_id": "CVE-2021-44228", "cik": "0000320193"}
    keywords: List[str] = field(default_factory=list)
    canonical_query: str = ""


# ---------------------------------------------------------------------------
# Pre-compiled Domain Knowledge Mappings
# ---------------------------------------------------------------------------
_FINANCIAL_ENTITIES: Dict[str, Dict[str, str]] = {
    "apple": {"name": "Apple Inc.", "ticker": "AAPL", "cik": "0000320193"},
    "aapl": {"name": "Apple Inc.", "ticker": "AAPL", "cik": "0000320193"},
    "tesla": {"name": "Tesla, Inc.", "ticker": "TSLA", "cik": "0001318605"},
    "tsla": {"name": "Tesla, Inc.", "ticker": "TSLA", "cik": "0001318605"},
    "microsoft": {"name": "Microsoft Corp", "ticker": "MSFT", "cik": "0000789019"},
    "msft": {"name": "Microsoft Corp", "ticker": "MSFT", "cik": "0000789019"},
    "amazon": {"name": "Amazon.com Inc", "ticker": "AMZN", "cik": "0001018724"},
    "amzn": {"name": "Amazon.com Inc", "ticker": "AMZN", "cik": "0001018724"},
    "google": {"name": "Alphabet Inc.", "ticker": "GOOGL", "cik": "0001652044"},
    "alphabet": {"name": "Alphabet Inc.", "ticker": "GOOGL", "cik": "0001652044"},
    "googl": {"name": "Alphabet Inc.", "ticker": "GOOGL", "cik": "0001652044"},
    "meta": {"name": "Meta Platforms, Inc.", "ticker": "META", "cik": "0001326801"},
    "facebook": {"name": "Meta Platforms, Inc.", "ticker": "META", "cik": "0001326801"},
    "nvidia": {"name": "NVIDIA Corp", "ticker": "NVDA", "cik": "0001045810"},
    "nvda": {"name": "NVIDIA Corp", "ticker": "NVDA", "cik": "0001045810"},
}

_DRUG_TERMS: List[str] = [
    "metformin", "aspirin", "ibuprofen", "pembrolizumab", "keytruda",
    "semaglutide", "ozempic", "wegovy", "insulin", "atorvastatin",
    "lipitor", "lisinopril", "amlodipine", "levothyroxine", "omeprazole",
    "doxycycline", "amoxicillin", "rituximab", "adalimumab", "humira"
]

_MEDICAL_CONDITIONS: List[str] = [
    "type 2 diabetes", "t2dm", "hypertension", "myocardial infarction",
    "colorectal cancer", "breast cancer", "lung cancer", "covid-19",
    "chronic kidney disease", "heart failure", "alzheimer's", "parkinson's",
    "multiple sclerosis", "rheumatoid arthritis", "asthma", "copd"
]

_CVE_REGEX = re.compile(r"\b(CVE-\d{4}-\d{4,7})\b", re.IGNORECASE)
_MITRE_TECHNIQUE_REGEX = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_CWE_REGEX = re.compile(r"\b(CWE-\d{1,5})\b", re.IGNORECASE)
_LEGAL_SECTION_REGEX = re.compile(r"\b(?:section|sec\.|article|art\.)\s*(\d+[A-Z]?)\b", re.IGNORECASE)


class EntityResolver:
    """Extracts, resolves, and canonicalizes entities from verification claims."""

    def resolve(self, claim_text: str, domain: str) -> EntityResolution:
        """
        Extract entities from claim text tailored to the target domain.

        Args:
            claim_text: The suspicious claim text.
            domain: Canonical domain string.

        Returns:
            EntityResolution dataclass containing extracted identifiers and canonical query.
        """
        domain_clean = (domain or "general").lower()
        res = EntityResolution(original_claim=claim_text, domain=domain_clean)

        # 1. Check Cybersecurity entities
        if domain_clean in ("cybersecurity", "nvd", "mitre", "cisa"):
            return self._resolve_cybersecurity(claim_text, res)

        # 2. Check Financial entities
        if domain_clean in ("finance", "economics", "sec"):
            return self._resolve_finance(claim_text, res)

        # 3. Check Healthcare / Biomedical entities
        if domain_clean in ("healthcare", "medicine", "pharmacy", "biology", "genetics"):
            return self._resolve_healthcare(claim_text, res)

        # 4. Check Legal entities
        if domain_clean in ("law", "legal_general", "government_public_policy"):
            return self._resolve_legal(claim_text, res)

        # Fallback / General entity resolution
        res.canonical_query = claim_text
        res.keywords = [w for w in claim_text.split() if len(w) > 3]
        return res

    # ------------------------------------------------------------------
    # Domain Resolvers
    # ------------------------------------------------------------------
    def _resolve_cybersecurity(self, claim_text: str, res: EntityResolution) -> EntityResolution:
        cve_matches = _CVE_REGEX.findall(claim_text)
        if cve_matches:
            cve_id = cve_matches[0].upper()
            res.primary_entity = cve_id
            res.entity_type = "CVE"
            res.identifiers["cve_id"] = cve_id
            res.keywords = [cve_id]
            res.canonical_query = cve_id
            return res

        mitre_matches = _MITRE_TECHNIQUE_REGEX.findall(claim_text)
        if mitre_matches:
            tech_id = mitre_matches[0].upper()
            res.primary_entity = tech_id
            res.entity_type = "MITRE_TECHNIQUE"
            res.identifiers["technique_id"] = tech_id
            res.keywords = [tech_id]
            res.canonical_query = tech_id
            return res

        cwe_matches = _CWE_REGEX.findall(claim_text)
        if cwe_matches:
            cwe_id = cwe_matches[0].upper()
            res.primary_entity = cwe_id
            res.entity_type = "CWE"
            res.identifiers["cwe_id"] = cwe_id
            res.keywords = [cwe_id]
            res.canonical_query = cwe_id
            return res

        # Extract threat / software names (e.g. Log4Shell, WannaCry)
        threat_terms = ["log4shell", "wannacry", "solarwinds", "lockbit", "apache", "log4j", "openssl"]
        for term in threat_terms:
            if re.search(rf"\b{term}\b", claim_text, re.IGNORECASE):
                res.primary_entity = term.capitalize()
                res.entity_type = "THREAT_NAME"
                res.identifiers["threat_name"] = term
                res.keywords = [term]
                res.canonical_query = f"{term} vulnerability"
                return res

        res.canonical_query = claim_text
        return res

    def _resolve_finance(self, claim_text: str, res: EntityResolution) -> EntityResolution:
        claim_lower = claim_text.lower()
        for key, info in _FINANCIAL_ENTITIES.items():
            if re.search(rf"\b{re.escape(key)}\b", claim_lower):
                res.primary_entity = info["name"]
                res.entity_type = "CORPORATE_ENTITY"
                res.identifiers["ticker"] = info["ticker"]
                res.identifiers["cik"] = info["cik"]
                res.identifiers["company_name"] = info["name"]
                res.keywords = [info["name"], info["ticker"], info["cik"]]
                res.canonical_query = f"{info['name']} {info['ticker']}"
                return res

        res.canonical_query = claim_text
        return res

    def _resolve_healthcare(self, claim_text: str, res: EntityResolution) -> EntityResolution:
        claim_lower = claim_text.lower()
        found_drugs = []
        found_conditions = []

        for drug in _DRUG_TERMS:
            if re.search(rf"\b{re.escape(drug)}\b", claim_lower):
                found_drugs.append(drug.capitalize())

        for condition in _MEDICAL_CONDITIONS:
            if re.search(rf"\b{re.escape(condition)}\b", claim_lower):
                found_conditions.append(condition.title())

        if found_drugs or found_conditions:
            primary = (found_drugs + found_conditions)[0]
            res.primary_entity = primary
            res.entity_type = "DRUG" if found_drugs else "CONDITION"
            if found_drugs:
                res.identifiers["drug_name"] = found_drugs[0]
            if found_conditions:
                res.identifiers["condition"] = found_conditions[0]
            res.keywords = found_drugs + found_conditions
            res.canonical_query = " ".join(found_drugs + found_conditions)
            return res

        res.canonical_query = claim_text
        return res

    def _resolve_legal(self, claim_text: str, res: EntityResolution) -> EntityResolution:
        sections = _LEGAL_SECTION_REGEX.findall(claim_text)
        if sections:
            sec_no = sections[0]
            res.identifiers["section"] = sec_no
            res.entity_type = "STATUTE_SECTION"

        legal_acts = ["IPC", "CrPC", "CPC", "BNS", "BNSS", "BSA", "GDPR", "Constitution", "IT Act"]
        for act in legal_acts:
            if re.search(rf"\b{re.escape(act)}\b", claim_text, re.IGNORECASE):
                res.primary_entity = act
                res.identifiers["act"] = act
                if "section" in res.identifiers:
                    res.canonical_query = f"Section {res.identifiers['section']} {act}"
                else:
                    res.canonical_query = act
                return res

        res.canonical_query = claim_text
        return res
