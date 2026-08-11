from __future__ import annotations

import logging
import re
from typing import List, Optional

from config.settings import get_settings
from schemas.models import Passage, AdapterMetadata
from utils.async_executor import gather_results
from utils.http_client import ResilientHttpClient, get_client
from claims.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


class HealthcareAdapter:
    def __init__(self) -> None:
        self.name = "healthcare"
        self.openfda_key = get_settings().openfda_key
        self.entity_resolver = EntityResolver()

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=["pubmed", "pubmed_central", "openfda", "clinicaltrials"],
            supports_live_search=True,
            cacheable=True,
            priority=10,
            max_results=50,
            is_stub=False,
        )

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("openfda"):
            return 0.98
        if source_id.startswith("pubmed") or source_id.startswith("pmc"):
            return 0.97
        if source_id.startswith("clinicaltrials"):
            return 0.95
        return 0.95

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        passages: List[Passage] = []
        try:
            client = get_client()

            # Resolve healthcare entities (drug names, medical conditions)
            resolution = self.entity_resolver.resolve(query, "healthcare")
            drug_name = resolution.identifiers.get("drug_name")

            search_query = self._sanitize_query_for_api(query, resolution)

            results = await gather_results([
                self._search_pubmed(client, search_query, k),
                self._search_pubmed_central(client, search_query, k),
                self._search_openfda(client, drug_name or search_query, query, k),
                self._search_clinicaltrials(client, search_query, k),
            ])

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Healthcare source search error: {result}")
                elif isinstance(result, list):
                    passages.extend(result)
        except Exception as e:
            logger.error(f"Failed healthcare search: {e}")

        return (
            sorted(passages, key=lambda x: x.relevance_score, reverse=True)[:k]
            if passages
            else passages
        )

    def _sanitize_query_for_api(
        self, query: str, resolution: Optional[Any] = None
    ) -> str:
        """Construct clean search terms using extracted entities or sanitized keywords."""
        if resolution and resolution.keywords:
            return " ".join(resolution.keywords[:4])

        cleaned = re.sub(r"[^\w\s]", " ", query)
        words = [
            w
            for w in cleaned.split()
            if len(w) > 1
            and w.lower()
            not in {
                "is",
                "a",
                "an",
                "the",
                "for",
                "and",
                "or",
                "of",
                "in",
                "to",
                "on",
                "with",
                "claim",
            }
        ]
        return " ".join(words[:5]) if words else query

    async def _search_pubmed(
        self, client: ResilientHttpClient, search_query: str, k: int
    ) -> List[Passage]:
        try:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": search_query,
                "retmode": "json",
                "retmax": k,
            }
            search_res = await client.get(
                search_url, adapter_name=self.name, params=params
            )

            ids = search_res.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            ids_str = ",".join(ids)
            
            # Fetch abstracts
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_res = await client.get(
                fetch_url,
                adapter_name=self.name,
                params={"db": "pubmed", "id": ids_str, "retmode": "xml", "rettype": "abstract"}
            )
            
            abstracts = {}
            if fetch_res and hasattr(fetch_res, "text") and fetch_res.text:
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(fetch_res.text)
                    for article in root.findall(".//PubmedArticle"):
                        pmid_elem = article.find(".//PMID")
                        if pmid_elem is not None:
                            pmid = pmid_elem.text
                            abs_texts = article.findall(".//AbstractText")
                            if abs_texts:
                                abstracts[pmid] = " ".join([elem.text for elem in abs_texts if elem.text])
                except ET.ParseError:
                    logger.warning("Failed to parse PubMed abstract XML")

            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_res = await client.get(
                summary_url,
                adapter_name=self.name,
                params={"db": "pubmed", "id": ids_str, "retmode": "json"},
            )

            result = summary_res.json().get("result", {})
            passages = []
            for pm_id in ids:
                if pm_id in result:
                    item = result[pm_id]
                    title = item.get("title", f"PubMed Article {pm_id}")
                    pub_date = item.get("pubdate", "unknown")
                    abstract_text = abstracts.get(pm_id, "")
                    snippet = abstract_text if abstract_text else f"{title}. Published in PubMed ({pub_date}). ID: {pm_id}."
                    passages.append(
                        Passage(
                            title=title,
                            source="pubmed",
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pm_id}/",
                            publication_date=pub_date,
                            snippet=snippet,
                            source_id=f"pubmed_{pm_id}",
                            relevance_score=0.5,
                        )
                    )
            return passages
        except Exception as e:
            logger.warning(f"PubMed search search failed. URL: unknown. Error: {e}")
            return []

    async def _search_pubmed_central(
        self, client: ResilientHttpClient, search_query: str, k: int
    ) -> List[Passage]:
        try:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_res = await client.get(
                search_url,
                adapter_name=self.name,
                params={
                    "db": "pmc",
                    "term": search_query,
                    "retmode": "json",
                    "retmax": k,
                },
            )

            ids = search_res.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            summary_res = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                adapter_name=self.name,
                params={"db": "pmc", "id": ",".join(ids), "retmode": "json"},
            )

            result = summary_res.json().get("result", {})
            passages = []
            for pmc_uid in ids:
                item = result.get(pmc_uid, {})
                if not item:
                    continue
                title = item.get("title", f"PubMed Central Article {pmc_uid}")
                pub_date = item.get("pubdate", "unknown")
                passages.append(
                    Passage(
                        title=title,
                        source="pubmed_central",
                        url=f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_uid}/",
                        publication_date=pub_date,
                        snippet=f"{title}. Full-text biomedical article indexed in PubMed Central ({pub_date}). PMCID: PMC{pmc_uid}.",
                        source_id=f"pmc_{pmc_uid}",
                        relevance_score=0.5,
                    )
                )
            return passages
        except Exception as e:
            logger.warning(f"PubMed Central search search failed. URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi. Error: {e}")
            return []

    async def _search_openfda(
        self, client: ResilientHttpClient, term: str, query: str, k: int
    ) -> List[Passage]:
        try:
            # Format openFDA search term properly
            clean_term = term.split()[0] if term.strip() else query.split()[0] if query.strip() else ""
            if not clean_term:
                logger.warning("Empty search term provided for OpenFDA search.")
                return []
            fda_query = f'openfda.generic_name:"{clean_term}" OR openfda.brand_name:"{clean_term}"'

            params = {"search": fda_query, "limit": k}
            if self.openfda_key:
                params["api_key"] = self.openfda_key
            res = await client.get(
                "https://api.fda.gov/drug/label.json",
                adapter_name=self.name,
                params=params,
            )

            data = res.json().get("results", [])
            passages = []
            for item in data:
                brand = item.get("openfda", {}).get(
                    "brand_name", [clean_term.capitalize()]
                )[0]
                desc_list = item.get("description", []) or item.get(
                    "indications_and_usage", []
                )
                desc = (
                    desc_list[0]
                    if desc_list
                    else "FDA approved drug label details."
                )
                if desc:
                    passages.append(
                        Passage(
                            title=f"FDA Drug Label: {brand}",
                            source="openfda",
                            url="https://api.fda.gov/drug/label.json",
                            publication_date="unknown",
                            snippet=f"FDA Official Label for {brand}: {desc[:400]}",
                            source_id=f"openfda_{brand.lower()}",
                            relevance_score=0.5,
                        )
                    )
            return passages
        except Exception as e:
            logger.warning(f"OpenFDA search search failed. URL: https://api.fda.gov/drug/label.json. Error: {e}")
            return []

    async def _search_clinicaltrials(
        self, client: ResilientHttpClient, search_query: str, k: int
    ) -> List[Passage]:
        try:
            res = await client.get(
                "https://clinicaltrials.gov/api/v2/studies",
                adapter_name=self.name,
                params={
                    "query.term": search_query,
                    "pageSize": k,
                    "format": "json",
                },
            )

            data = res.json().get("studies", [])
            passages = []
            for study in data:
                protocol = study.get("protocolSection", {})
                nctId = protocol.get("identificationModule", {}).get("nctId", "")
                title = protocol.get("identificationModule", {}).get(
                    "briefTitle", "Clinical Study"
                )
                status = protocol.get("statusModule", {}).get(
                    "overallStatus", "Completed"
                )
                summary = protocol.get("descriptionModule", {}).get(
                    "briefSummary", ""
                )

                passages.append(
                    Passage(
                        title=f"Clinical Trial [{nctId}]: {title}",
                        source="clinicaltrials",
                        url=f"https://clinicaltrials.gov/study/{nctId}",
                        publication_date="unknown",
                        snippet=f"Clinical Study [{nctId}] - Status [{status}]: {title}. {summary[:350]}",
                        source_id=f"clinicaltrials_{nctId}",
                        relevance_score=0.5,
                    )
                )
            return passages
        except Exception as e:
            logger.warning(f"ClinicalTrials search search failed. URL: https://clinicaltrials.gov/api/v2/studies. Error: {e}")
            return []
