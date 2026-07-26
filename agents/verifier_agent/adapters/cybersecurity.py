from __future__ import annotations

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional

import httpx

from schemas.models import Passage, AdapterMetadata

logger = logging.getLogger(__name__)

class CybersecurityAdapter:
    def __init__(self) -> None:
        self.name = "cybersecurity"
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

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    self._search_nvd(client, query, k),
                    self._search_mitre(client, query, k),
                    self._search_cisa(client, query, k),
                    return_exceptions=True
                )
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Cybersecurity source search error: {result}")
                    elif isinstance(result, list):
                        passages.extend(result)
        except Exception as e:
            logger.error(f"Failed cybersecurity search: {e}")
            
        return sorted(passages, key=lambda x: x.relevance_score, reverse=True)[:k] if passages else passages

    async def _search_nvd(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={query}&resultsPerPage={k}"
            res = await client.get(url, timeout=15.0)
            res.raise_for_status()
            
            data = res.json().get("vulnerabilities", [])
            passages = []
            for item in data:
                cve = item.get("cve", {})
                cve_id = cve.get("id", "")
                descriptions = cve.get("descriptions", [])
                desc_text = descriptions[0].get("value", "") if descriptions else ""
                published = cve.get("published", "")
                
                if desc_text:
                    passages.append(Passage(
                        text=f"{cve_id} (Published: {published}): {desc_text}",
                        source_id=f"nvd_{cve_id}",
                        source_url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                        relevance_score=0.9
                    ))
            return passages
        except Exception as e:
            logger.error(f"NVD search error: {e}")
            return []

    async def _search_mitre(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            if not self._mitre_cache:
                url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
                res = await client.get(url, timeout=20.0)
                res.raise_for_status()
                objects = res.json().get("objects", [])
                self._mitre_cache = [
                    obj for obj in objects 
                    if obj.get("type") in ["attack-pattern", "malware", "tool"]
                ]
            
            passages = []
            query_lower = query.lower()
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
                        text=f"MITRE ATT&CK: {obj.get('name')}\n{obj.get('description', '')}",
                        source_id=f"mitre_{ext_id}",
                        source_url=f"https://attack.mitre.org/techniques/{ext_id}",
                        relevance_score=0.85
                    ))
                    if len(passages) >= k:
                        break
            return passages
        except Exception as e:
            logger.error(f"MITRE search error: {e}")
            return []

    async def _search_cisa(self, client: httpx.AsyncClient, query: str, k: int) -> List[Passage]:
        try:
            if not self._cisa_cache:
                url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
                res = await client.get(url, timeout=10.0)
                res.raise_for_status()
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
                        cve = item.get("cveID", "")
                        passages.append(Passage(
                            text=f"CISA KEV {cve}: {item.get('vendorProject')} {item.get('product')}\n{item.get('shortDescription')}",
                            source_id=f"cisa_{cve}",
                            source_url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
                            relevance_score=0.95
                        ))
                        if len(passages) >= k:
                            break
            return passages
        except Exception as e:
            logger.error(f"CISA search error: {e}")
            return []

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("nvd"): return 0.95
        if source_id.startswith("mitre"): return 0.98
        if source_id.startswith("cisa"): return 0.99
        return 0.85
