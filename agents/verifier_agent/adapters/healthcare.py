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
        self.sources_attempted = ["pubmed", "pubmed_central", "openfda", "clinicaltrials"]
        self.sources_succeeded = []
        self.sources_failed = []

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

            source_names = ["pubmed", "pubmed_central", "openfda", "clinicaltrials"]
            for s_name, result in zip(source_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Healthcare source {s_name} search error: {result}")
                    self.sources_failed.append(s_name)
                elif isinstance(result, list) and result:
                    passages.extend(result)
                    self.sources_succeeded.append(s_name)
        except Exception as e:
            logger.error(f"Failed healthcare search: {e}")

        return self._select_source_diverse_passages(passages, k)

    @staticmethod
    def _select_source_diverse_passages(passages: List[Passage], k: int) -> List[Passage]:
        """Retain the strongest record from each source before filling remaining slots."""
        if k <= 0 or not passages:
            return []
        ranked = sorted(passages, key=lambda passage: passage.relevance_score, reverse=True)
        selected: List[Passage] = []
        selected_sources: set[str] = set()
        for passage in ranked:
            if passage.source not in selected_sources:
                selected.append(passage)
                selected_sources.add(passage.source)
                if len(selected) == k:
                    return selected
        selected_ids = {id(passage) for passage in selected}
        selected.extend(passage for passage in ranked if id(passage) not in selected_ids)
        return selected[:k]

    def _sanitize_query_for_api(
        self, query: str, resolution: Optional[Any] = None
    ) -> str:
        """Construct clean search terms using extracted entities or sanitized keywords."""
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
        contextual_terms = words[:5]
        if resolution and resolution.keywords:
            resolved_terms = list(resolution.keywords[:4])
            resolved_lower = {term.lower() for term in resolved_terms}
            contextual_terms = [
                term for term in contextual_terms if term.lower() not in resolved_lower
            ]
            return " ".join((resolved_terms + contextual_terms)[:6])
        return " ".join(contextual_terms) if contextual_terms else query

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
                    snippet = abstract_text if len(abstract_text) > 30 else f"Title: {title}. PubMed Article ({pub_date}) ID: {pm_id}."
                    passages.append(
                        Passage(
                            title=title,
                            source="pubmed",
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pm_id}/",
                            publication_date=pub_date,
                            snippet=snippet[:450],
                            source_id=f"pubmed_{pm_id}",
                            relevance_score=0.75,
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

            ids_str = ",".join(ids)

            summary_res = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                adapter_name=self.name,
                params={"db": "pmc", "id": ids_str, "retmode": "json"},
            )

            result = summary_res.json().get("result", {})

            # Summary is fetched first so metadata remains available if the
            # full-text endpoint is slow or temporarily unavailable.
            pmc_abstracts = {}
            try:
                fetch_res = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    adapter_name=self.name,
                    params={"db": "pmc", "id": ids_str, "retmode": "xml"},
                )
                if fetch_res and getattr(fetch_res, "text", ""):
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(fetch_res.text)
                    for article in root.findall(".//article"):
                        id_elem = article.find(".//article-id[@pub-id-type='pmc']")
                        pmc_id = id_elem.text if id_elem is not None else None
                        if pmc_id:
                            paragraphs = article.findall(".//abstract//p") or article.findall(".//body//p")
                            text = " ".join(p.text for p in paragraphs if p.text)
                            if text:
                                pmc_abstracts[pmc_id] = text
            except Exception as exc:
                logger.warning("PMC full-text extraction failed: %s", exc)
            passages = []
            for pmc_uid in ids:
                item = result.get(pmc_uid, {})
                if not item:
                    continue
                title = item.get("title", f"PubMed Central Article {pmc_uid}")
                pub_date = item.get("pubdate", "unknown")
                abs_text = pmc_abstracts.get(pmc_uid, "")
                snippet = abs_text if len(abs_text) > 30 else f"Article [{title}]: PMC full-text study ({pub_date})."
                passages.append(
                    Passage(
                        title=title,
                        source="pubmed_central",
                        url=f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_uid}/",
                        publication_date=pub_date,
                        snippet=snippet[:450],
                        source_id=f"pmc_{pmc_uid}",
                        relevance_score=0.75,
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
