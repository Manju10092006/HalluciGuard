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
        return 0.96

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            client = get_client()

            # Resolve entity to check for exact CVE ID
            resolution = self.entity_resolver.resolve(query, "cybersecurity")
            cve_id = resolution.identifiers.get("cve_id")

            results = await gather_results([
                self._search_nvd(client, query, cve_id, k),
                self._search_mitre(client, query, cve_id, k),
                self._search_cisa(client, query, cve_id, k),
            ])

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Cybersecurity source search error: {result}")
                elif isinstance(result, list):
                    passages.extend(result)
        except Exception as e:
            logger.error(f"Failed cybersecurity search: {e}")

        return (
            sorted(passages, key=lambda x: x.relevance_score, reverse=True)[:k]
            if passages
            else passages
        )

    async def _search_nvd(
        self,
        client: ResilientHttpClient,
        query: str,
        cve_id: Optional[str],
        k: int,
    ) -> List[Passage]:
        try:
            headers = {"apiKey": self.nvd_api_key} if self.nvd_api_key else None
            url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

            # 1. Primary Search: If explicit CVE ID detected, query using cveId parameter
            if cve_id:
                try:
                    res = await client.get(
                        url,
                        adapter_name=self.name,
                        params={"cveId": cve_id},
                        headers=headers,
                    )
                    data = res.json().get("vulnerabilities", [])
                    passages = self._parse_nvd_vulnerabilities(data)
                    if passages:
                        return passages
                except Exception as ex:
                    logger.warning(f"NVD cveId lookup for {cve_id} failed: {ex}")

            # 2. Fallback Keyword Search using entity keyword instead of long claim sentence
            cve_match = _CVE_REGEX.findall(query)
            search_term = cve_match[0] if cve_match else (cve_id or query)

            # Keep search_term to entity keyword or short string to prevent NVD API empty results
            if len(search_term.split()) > 4 and not cve_match:
                search_term = " ".join([w for w in search_term.split() if len(w) > 3][:3])

            res = await client.get(
                url,
                adapter_name=self.name,
                params={"keywordSearch": search_term, "resultsPerPage": k},
                headers=headers,
            )
            data = res.json().get("vulnerabilities", [])
            return self._parse_nvd_vulnerabilities(data)
        except Exception as e:
            logger.warning(f"NVD search search failed. URL: unknown. Error: {e}")
            return []

    def _parse_nvd_vulnerabilities(self, data: List[Dict[str, Any]]) -> List[Passage]:
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
                        source_id=f"nvd_{cve_id}",
                        relevance_score=0.5,
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
                res = await client.get(url, adapter_name=self.name)
                objects = res.json().get("objects", [])
                self._mitre_cache = [
                    obj
                    for obj in objects
                    if obj.get("type") in ["attack-pattern", "malware", "tool"]
                ]

            passages = []
            search_terms = []
            if cve_id:
                search_terms.append(cve_id.lower())
            search_terms.extend([w.lower() for w in query.split() if len(w) > 3])

            if self._mitre_cache:
                for obj in self._mitre_cache:
                    name = obj.get("name", "").lower()
                    desc = obj.get("description", "").lower()

                    # Match CVE ID or keyword terms
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
                                source_id=f"mitre_{ext_id}",
                                relevance_score=0.5,
                            )
                        )
                        if len(passages) >= k:
                            break
            return passages
        except Exception as e:
            logger.warning(f"MITRE search search failed. URL: unknown. Error: {e}")
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
                res = await client.get(url, adapter_name=self.name)
                self._cisa_cache = res.json().get("vulnerabilities", [])

            passages = []
            search_term = (cve_id or query).lower()

            if self._cisa_cache:
                for item in self._cisa_cache:
                    cve_val = str(item.get("cveID", "")).lower()
                    vendor_val = str(item.get("vendorProject", "")).lower()
                    product_val = str(item.get("product", "")).lower()
                    desc_val = str(item.get("shortDescription", "")).lower()

                    if (
                        cve_id and cve_id.lower() in cve_val
                    ) or (
                        search_term in cve_val
                        or search_term in vendor_val
                        or search_term in product_val
                        or search_term in desc_val
                    ):
                        cve = item.get("cveID", "CVE")
                        passages.append(
                            Passage(
                                title=f"CISA KEV: {cve}",
                                source="cisa",
                                url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
                                publication_date=str(
                                    item.get("dateAdded", "unknown")
                                )[:10],
                                snippet=f"CISA KEV [{cve}] {item.get('vendorProject')} {item.get('product')}: {item.get('shortDescription', '')[:350]}",
                                source_id=f"cisa_{cve}",
                                relevance_score=0.5,
                            )
                        )
                        if len(passages) >= k:
                            break
            return passages
        except Exception as e:
            logger.warning(f"CISA search search failed. URL: unknown. Error: {e}")
            return []
