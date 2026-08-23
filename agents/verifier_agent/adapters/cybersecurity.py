from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

from config.settings import get_settings
from schemas.models import Passage, AdapterMetadata
from utils.async_executor import gather_results
from utils.http_client import ResilientHttpClient, get_client
from claims.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)

_CVE_REGEX = re.compile(r"\b(CVE-\d{4}-\d{4,7})\b", re.IGNORECASE)


class CybersecurityAdapter:
    def __init__(self) -> None:
        self.name = "cybersecurity"
        self.nvd_api_key = get_settings().nvd_api_key
        self._mitre_cache: Optional[List[Dict[str, Any]]] = None
        self._cisa_cache: Optional[List[Dict[str, Any]]] = None
        self.entity_resolver = EntityResolver()

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["nvd", "mitre", "cisa"],
            supports_live_search=True,
            cacheable=True,
            priority=10,
            max_results=50,
            is_stub=False,
        )

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("nvd"):
            return 0.99
        if source_id.startswith("cisa"):
            return 0.98
        if source_id.startswith("circl"):
            return 0.97
        if source_id.startswith("mitre"):
            return 0.97
        if source_id.startswith("github_advisory"):
            return 0.96
        return 0.95

    async def search(
        self,
        query: str,
        k: int = 5,
        source_mode: Optional[str] = None,
        **kwargs,
    ) -> List[Passage]:
        passages: List[Passage] = []
        try:
            client = get_client()

            # Extract CVE ID directly via regex or entity resolver
            cve_matches = _CVE_REGEX.findall(query)
            resolution = self.entity_resolver.resolve(query, "cybersecurity")
            cve_id = cve_matches[0].upper() if cve_matches else resolution.identifiers.get("cve_id")

            # Handle source_mode override
            if source_mode:
                mode_clean = source_mode.lower().strip()
                if mode_clean in ("cybersecurity-nvd", "nvd"):
                    return await self._search_nvd(client, query, cve_id, k)
                if mode_clean in ("cybersecurity-cisa", "cisa"):
                    return await self._search_cisa(client, query, cve_id, k)
                if mode_clean in ("cybersecurity-mitre", "mitre"):
                    return await self._search_mitre(client, query, cve_id, k)

            tasks = []
            if cve_id:
                tasks.append(self._search_cve_direct(client, cve_id))
                tasks.append(self._search_cisa(client, query, cve_id, k))
            else:
                tasks.append(self._search_nvd(client, query, cve_id, k))
                tasks.append(self._search_cisa(client, query, cve_id, k))
                tasks.append(self._search_mitre(client, query, cve_id, k))

            results = await gather_results(tasks)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Cybersecurity source search error: {result}")
                elif isinstance(result, list):
                    passages.extend(result)
        except Exception as e:
            logger.error(f"Failed cybersecurity search: {e}")

        # Deduplicate passages by URL
        seen_urls = set()
        deduped = []
        for p in passages:
            key = p.url or p.source_id
            if key not in seen_urls:
                seen_urls.add(key)
                deduped.append(p)

        return (
            sorted(deduped, key=lambda x: x.relevance_score, reverse=True)[:k]
            if deduped
            else deduped
        )

    async def _search_cve_direct(
        self,
        client: ResilientHttpClient,
        cve_id: str,
    ) -> List[Passage]:
        """Direct, high-priority lookup for a specific CVE ID across NIST NVD, CIRCL, and GitHub."""
        passages: List[Passage] = []
        cve_clean = cve_id.strip().upper()

        # 1. Try NIST NVD
        try:
            headers = {"apiKey": self.nvd_api_key} if self.nvd_api_key else None
            res = await client.get(
                "https://services.nvd.nist.gov/rest/json/cves/2.0",
                adapter_name=self.name,
                params={"cveId": cve_clean},
                headers=headers,
            )
            data = res.json().get("vulnerabilities", [])
            nvd_passages = self._parse_nvd_vulnerabilities(data, relevance_score=0.85)
            if nvd_passages:
                return nvd_passages
        except Exception as ex:
            logger.warning(f"NVD direct lookup for {cve_clean} failed: {ex}")

        # 2. Try CIRCL CVE API (Fast public CVE database)
        try:
            res = await client.get(
                f"https://cve.circl.lu/api/cve/{cve_clean}",
                adapter_name=self.name,
            )
            cve_data = res.json()
            if cve_data and isinstance(cve_data, dict) and cve_data.get("id"):
                summary = cve_data.get("summary") or cve_data.get("description", "")
                pub_date = str(cve_data.get("Published", "unknown"))[:10]
                passages.append(
                    Passage(
                        title=f"CIRCL CVE: {cve_clean}",
                        source="circl",
                        url=f"https://nvd.nist.gov/vuln/detail/{cve_clean}",
                        publication_date=pub_date,
                        snippet=f"Vulnerability {cve_clean}: {summary[:400]}",
                        source_id=f"circl_{cve_clean.lower()}",
                        relevance_score=0.0,
                        source_confidence_hint=0.85,
                    )
                )
                return passages
        except Exception as ex:
            logger.warning(f"CIRCL lookup for {cve_clean} failed: {ex}")

        # 3. Try GitHub Advisory Database
        try:
            res = await client.get(
                "https://api.github.com/advisories",
                adapter_name=self.name,
                params={"cve_id": cve_clean},
            )
            advisories = res.json()
            if isinstance(advisories, list) and advisories:
                adv = advisories[0]
                summary = adv.get("summary") or adv.get("description", "")
                ghsa_id = adv.get("ghsa_id", cve_clean)
                passages.append(
                    Passage(
                        title=f"GitHub Advisory: {ghsa_id} ({cve_clean})",
                        source="github_advisory",
                        url=adv.get("html_url") or f"https://github.com/advisories/{ghsa_id}",
                        publication_date=str(adv.get("published_at", "unknown"))[:10],
                        snippet=f"Security Advisory for {cve_clean} [{ghsa_id}]: {summary[:400]}",
                        source_id=f"ghsa_{ghsa_id.lower()}",
                        relevance_score=0.0,
                        source_confidence_hint=0.85,
                    )
                )
                return passages
        except Exception as ex:
            logger.warning(f"GitHub advisory lookup for {cve_clean} failed: {ex}")

        return passages

    async def _search_nvd(
        self,
        client: ResilientHttpClient,
        query: str,
        cve_id: Optional[str],
        k: int,
    ) -> List[Passage]:
        try:
            if cve_id:
                direct = await self._search_cve_direct(client, cve_id)
                if direct:
                    return direct

            headers = {"apiKey": self.nvd_api_key} if self.nvd_api_key else None
            url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

            # Clean search terms to entity keywords
            search_terms = [w for w in query.split() if len(w) > 3 and not w.lower() in ("most", "dangerous", "cyber", "attack", "virus")]
            search_term = " ".join(search_terms[:3]) if search_terms else query[:50]

            res = await client.get(
                url,
                adapter_name=self.name,
                params={"keywordSearch": search_term, "resultsPerPage": k},
                headers=headers,
            )
            data = res.json().get("vulnerabilities", [])
            return self._parse_nvd_vulnerabilities(data)
        except Exception as e:
            logger.warning(f"NVD keyword search failed: {e}")
            return []

    def _parse_nvd_vulnerabilities(
        self,
        data: List[Dict[str, Any]],
        relevance_score: float = 0.0,
        source_confidence_hint: float = 0.80,
    ) -> List[Passage]:
        passages = []
        for item in data:
            cve = item.get("cve", {})
            cve_id = cve.get("id", "CVE-Unknown")
            descriptions = cve.get("descriptions", [])
            desc_text = descriptions[0].get("value", "") if descriptions else ""
            published = cve.get("published", "unknown")

            if desc_text:
                passages.append(
                    Passage(
                        title=f"NVD CVE: {cve_id}",
                        source="nvd",
                        url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                        publication_date=str(published)[:10],
                        snippet=f"Vulnerability {cve_id}: {desc_text[:400]}",
                        source_id=f"nvd_{cve_id.lower().replace('-', '_')}",
                        relevance_score=relevance_score,
                        source_confidence_hint=source_confidence_hint,
                    )
                )
        return passages

    def _mitre_attack_url(self, external_id: str) -> str:
        if "." in external_id:
            technique, subtechnique = external_id.split(".", 1)
            return f"https://attack.mitre.org/techniques/{technique}/{subtechnique}/"
        return f"https://attack.mitre.org/techniques/{external_id}/"

    async def _search_mitre(
        self,
        client: ResilientHttpClient,
        query: str,
        cve_id: Optional[str],
        k: int,
    ) -> List[Passage]:
        try:
            if not self._mitre_cache:
                url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
                try:
                    res = await client.get(url, adapter_name=self.name)
                    objects = res.json().get("objects", [])
                    self._mitre_cache = [
                        obj
                        for obj in objects
                        if obj.get("type") in ["attack-pattern", "malware", "tool"]
                    ]
                except Exception as ex:
                    logger.warning(f"Failed to fetch MITRE STIX JSON: {ex}")
                    self._mitre_cache = []

            passages = []
            stopwords = {"most", "dangerous", "cyber", "attack", "is", "the", "a", "an", "in", "of", "to"}
            search_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3 and w.lower() not in stopwords]
            if cve_id:
                search_terms.insert(0, cve_id.lower())

            if self._mitre_cache:
                for obj in self._mitre_cache:
                    name = obj.get("name", "").lower()
                    desc = obj.get("description", "").lower()

                    is_match = any(st in name or st in desc for st in search_terms if len(st) > 3)
                    if is_match:
                        ext_id = ""
                        for ext_ref in obj.get("external_references", []):
                            if ext_ref.get("source_name") == "mitre-attack":
                                ext_id = ext_ref.get("external_id", "")
                                break
                        passages.append(
                            Passage(
                                title=f"MITRE ATT&CK: {obj.get('name')}",
                                source="mitre",
                                url=self._mitre_attack_url(ext_id)
                                if ext_id
                                else "https://attack.mitre.org/",
                                publication_date=str(
                                    obj.get("modified", obj.get("created", "unknown"))
                                )[:10],
                                snippet=f"Technique [{ext_id}]: {obj.get('description', '')[:350]}",
                                source_id=f"mitre_{ext_id.lower().replace('.', '_')}",
                                relevance_score=0.0,
                                source_confidence_hint=0.75,
                            )
                        )
                        if len(passages) >= k:
                            break
            return passages
        except Exception as e:
            logger.warning(f"MITRE search failed: {e}")
            return []

    async def _search_cisa(
        self,
        client: ResilientHttpClient,
        query: str,
        cve_id: Optional[str],
        k: int,
    ) -> List[Passage]:
        try:
            if not self._cisa_cache:
                url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
                try:
                    res = await client.get(url, adapter_name=self.name)
                    self._cisa_cache = res.json().get("vulnerabilities", [])
                except Exception as ex:
                    logger.warning(f"Failed to fetch CISA KEV feed: {ex}")
                    self._cisa_cache = []

            passages = []
            stopwords = {"is", "the", "a", "an", "in", "of", "to", "and", "or", "for", "with", "associated", "related", "cve", "vulnerability", "flaw", "attack", "exploit"}
            query_terms = [
                w.lower() for w in re.findall(r"\w+", query)
                if len(w) > 2 and w.lower() not in stopwords and not (w.isdigit() and len(w) == 4)
            ]

            if self._cisa_cache:
                scored_candidates = []
                for item in self._cisa_cache:
                    cve_val = str(item.get("cveID", "")).lower().strip()
                    vendor_val = str(item.get("vendorProject", "")).lower()
                    product_val = str(item.get("product", "")).lower()
                    vuln_name = str(item.get("vulnerabilityName", "")).lower()
                    desc_val = str(item.get("shortDescription", "")).lower()
                    combined_text = f"{cve_val} {vendor_val} {product_val} {vuln_name} {desc_val}"

                    match_score = 0
                    if cve_id and cve_id.lower().strip() == cve_val:
                        match_score = 100
                    elif not cve_id and query_terms:
                        matches = sum(1 for t in query_terms if t in combined_text)
                        if matches >= max(1, len(query_terms) // 2):
                            match_score = matches

                    if match_score > 0:
                        scored_candidates.append((match_score, item))

                # Sort by match score descending
                scored_candidates.sort(key=lambda x: x[0], reverse=True)

                for _, item in scored_candidates[:k]:
                    cve = item.get("cveID", "CVE")
                    vendor = item.get("vendorProject", "")
                    product = item.get("product", "")
                    vuln = item.get("vulnerabilityName", "")
                    desc = item.get("shortDescription", "")
                    passages.append(
                        Passage(
                            title=f"CISA KEV: {cve} - {product}",
                            source="cisa",
                            url=f"https://www.cisa.gov/known-exploited-vulnerabilities-catalog?search_api_fulltext={cve}",
                            publication_date=str(item.get("dateAdded", "unknown"))[:10],
                            snippet=f"CISA Known Exploited Vulnerability [{cve}] {vendor} {product} ({vuln}): {desc[:350]}",
                            source_id=f"cisa_{cve.lower().replace('-', '_')}",
                            relevance_score=0.0,
                            source_confidence_hint=0.85,
                        )
                    )
            return passages
        except Exception as e:
            logger.warning(f"CISA search failed: {e}")
            return []
