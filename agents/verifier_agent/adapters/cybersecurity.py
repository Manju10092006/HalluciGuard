from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from config.settings import get_settings
from schemas.models import Passage, AdapterMetadata
from utils.async_executor import gather_results
from utils.http_client import ResilientHttpClient, get_client

logger = logging.getLogger(__name__)

class CybersecurityAdapter:
    def __init__(self) -> None:
        self.name = "cybersecurity"
        self.nvd_api_key = get_settings().nvd_api_key
        self._mitre_cache: Optional[List[Dict[str, Any]]] = None
        self._cisa_cache: Optional[List[Dict[str, Any]]] = None

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
            is_stub=False
        )

    def credibility_of(self, source_id: str) -> float:
        return 0.96

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            client = get_client()
            results = await gather_results([
                self._search_nvd(client, query, k),
                self._search_mitre(client, query, k),
                self._search_cisa(client, query, k),
            ])
                
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Cybersecurity source search error: {result}")
                elif isinstance(result, list):
                    passages.extend(result)
        except Exception as e:
            logger.error(f"Failed cybersecurity search: {e}")
            
        return sorted(passages, key=lambda x: x.relevance_score, reverse=True)[:k] if passages else passages

    async def _search_nvd(self, client: ResilientHttpClient, query: str, k: int) -> List[Passage]:
        try:
            headers = {"apiKey": self.nvd_api_key} if self.nvd_api_key else None
            res = await client.get(
                "https://services.nvd.nist.gov/rest/json/cves/2.0",
                adapter_name=self.name,
                params={"keywordSearch": query, "resultsPerPage": k},
                headers=headers,
            )
            
            data = res.json().get("vulnerabilities", [])
            passages = []
            for item in data:
                cve = item.get("cve", {})
                cve_id = cve.get("id", "CVE-Unknown")
                descriptions = cve.get("descriptions", [])
                desc_text = descriptions[0].get("value", "") if descriptions else ""
                published = cve.get("published", "2023")
                
                if desc_text:
                    passages.append(Passage(
                        title=f"NVD CVE: {cve_id}",
                        source="nvd",
                        url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                        publication_date=str(published)[:10],
                        snippet=f"Vulnerability {cve_id}: {desc_text[:300]}",
                        source_id=f"nvd_{cve_id}",
                        relevance_score=0.9
                    ))
            return passages
        except Exception as e:
            logger.error(f"NVD search error: {e}")
            return []

    def _mitre_attack_url(self, external_id: str) -> str:
        if "." in external_id:
            technique, subtechnique = external_id.split(".", 1)
            return f"https://attack.mitre.org/techniques/{technique}/{subtechnique}/"
        return f"https://attack.mitre.org/techniques/{external_id}/"

    async def _search_mitre(self, client: ResilientHttpClient, query: str, k: int) -> List[Passage]:
        try:
            if not self._mitre_cache:
                url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
                res = await client.get(url, adapter_name=self.name)
                objects = res.json().get("objects", [])
                self._mitre_cache = [
                    obj for obj in objects 
                    if obj.get("type") in ["attack-pattern", "malware", "tool"]
                ]
            
            passages = []
            query_lower = query.lower()
            if self._mitre_cache:
                for obj in self._mitre_cache:
                    name = obj.get("name", "").lower()
                    desc = obj.get("description", "").lower()
                    if query_lower in name or query_lower in desc:
                        ext_id = ""
                        for ext_ref in obj.get("external_references", []):
                            if ext_ref.get("source_name") == "mitre-attack":
                                ext_id = ext_ref.get("external_id", "")
                                break
                        passages.append(Passage(
                            title=f"MITRE ATT&CK: {obj.get('name')}",
                            source="mitre",
                            url=self._mitre_attack_url(ext_id) if ext_id else "https://attack.mitre.org/",
                            publication_date=str(obj.get("modified", obj.get("created", "2024")))[:10],
                            snippet=f"Technique [{ext_id}]: {obj.get('description', '')[:300]}",
                            source_id=f"mitre_{ext_id}",
                            relevance_score=0.85
                        ))
                        if len(passages) >= k:
                            break
            return passages
        except Exception as e:
            logger.error(f"MITRE search error: {e}")
            return []

    async def _search_cisa(self, client: ResilientHttpClient, query: str, k: int) -> List[Passage]:
        try:
            if not self._cisa_cache:
                url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
                res = await client.get(url, adapter_name=self.name)
                self._cisa_cache = res.json().get("vulnerabilities", [])
                
            passages = []
            query_lower = query.lower()
            if self._cisa_cache:
                for item in self._cisa_cache:
                    match_fields = [
                        item.get("cveID", ""),
                        item.get("vendorProject", ""),
                        item.get("product", ""),
                        item.get("shortDescription", "")
                    ]
                    if any(query_lower in str(field).lower() for field in match_fields):
                        cve = item.get("cveID", "CVE")
                        passages.append(Passage(
                            title=f"CISA KEV: {cve}",
                            source="cisa",
                            url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
                            publication_date=str(item.get("dateAdded", "2024"))[:10],
                            snippet=f"CISA KEV [{cve}] {item.get('vendorProject')} {item.get('product')}: {item.get('shortDescription', '')[:300]}",
                            source_id=f"cisa_{cve}",
                            relevance_score=0.88
                        ))
                        if len(passages) >= k:
                            break
            return passages
        except Exception as e:
            logger.error(f"CISA search error: {e}")
            return []
