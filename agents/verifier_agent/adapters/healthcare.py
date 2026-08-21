from __future__ import annotations

import logging
import re
from typing import List, Optional, Any, Dict

from config.settings import get_settings
from schemas.models import Passage, AdapterMetadata
from utils.async_executor import gather_results
from utils.http_client import ResilientHttpClient, get_client
from claims.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


class HealthcareAdapter:
    """
    Authoritative Healthcare Adapter (V1.3).
    Integrates PubMed, PubMed Central (PMC), OpenFDA, ClinicalTrials.gov, and WHO.
    Supports query strategy routing and granular diagnostic source modes.
    """

    def __init__(self) -> None:
        self.name = "healthcare"
        self.openfda_key = get_settings().openfda_key
        self.entity_resolver = EntityResolver()
        self.sources_attempted: List[str] = []
        self.sources_succeeded: List[str] = []
        self.sources_failed: List[str] = []

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.3.0",
            supported_domains=["pubmed", "pubmed_central", "openfda", "clinicaltrials", "who"],
            supports_live_search=True,
            cacheable=True,
            priority=10,
            max_results=50,
            is_stub=False,
        )

    def credibility_of(self, source_id: str) -> float:
        if source_id.startswith("who"):
            return 0.99
        if source_id.startswith("openfda"):
            return 0.98
        if source_id.startswith("pubmed") or source_id.startswith("pmid"):
            return 0.97
        if source_id.startswith("pmc"):
            return 0.96
        if source_id.startswith("clinicaltrials"):
            return 0.95
        return 0.95

    def _determine_query_strategy(self, query: str, drug_name: Optional[str] = None) -> List[str]:
        """Determine which sources to query based on claim terminology and entity resolution."""
        q_lower = query.lower()
        drug_terms = {
            "cure", "cures", "treat", "treats", "treatment", "tablet", "tablets", "capsule",
            "drug", "drugs", "medicine", "medication", "pill", "pills", "dose", "dosage",
            "side effect", "adverse", "fda", "approved", "therapy", "antibiotic",
        }
        trial_terms = {
            "trial", "trials", "clinical", "study", "studies", "efficacy", "safety",
            "phase 1", "phase 2", "phase 3", "placebo", "cohort", "randomized",
        }
        public_health_terms = {
            "outbreak", "pandemic", "epidemic", "mortality", "prevalence", "incidence",
            "vaccination", "vaccine", "global health", "who", "guideline", "guidelines",
            "eradication", "transmission", "virus", "infection",
        }

        has_drug = bool(drug_name) or any(t in q_lower for t in drug_terms)
        has_trial = any(t in q_lower for t in trial_terms)
        has_pub_health = any(t in q_lower for t in public_health_terms)

        sources = []
        if has_drug:
            sources.extend(["openfda", "pubmed", "pubmed_central"])
        if has_trial:
            sources.extend(["clinicaltrials", "pubmed", "pubmed_central"])
        if has_pub_health:
            sources.extend(["who", "pubmed"])

        # Fallback to all standard sources if ambiguous
        if not sources:
            sources = ["pubmed", "openfda", "pubmed_central", "clinicaltrials", "who"]

        # Deduplicate while preserving order
        seen = set()
        return [s for s in sources if not (s in seen or seen.add(s))]

    async def search(
        self,
        query: str,
        k: int = 5,
        source_mode: Optional[str] = None,
        retrieval_mode: str = "hybrid",
    ) -> List[Passage]:
        """
        Search authoritative medical and health databases.

        Args:
            query: Claim or search text.
            k: Max results to return.
            source_mode: Diagnostic source override ('healthcare-pubmed', 'healthcare-fda', etc.)
            retrieval_mode: Standard pipeline retrieval mode parameter.
        """
        passages: List[Passage] = []
        client = get_client()

        # Resolve healthcare entities
        resolution = self.entity_resolver.resolve(query, "healthcare")
        drug_name = resolution.identifiers.get("drug_name") if resolution else None

        search_query = self._sanitize_query_for_api(query, resolution)

        # Source selection based on mode or strategy
        if source_mode == "healthcare-pubmed":
            active_sources = ["pubmed"]
        elif source_mode == "healthcare-pmc":
            active_sources = ["pubmed_central"]
        elif source_mode == "healthcare-fda":
            active_sources = ["openfda"]
        elif source_mode == "healthcare-who":
            active_sources = ["who"]
        elif source_mode == "healthcare-clinicaltrials":
            active_sources = ["clinicaltrials"]
        else:
            active_sources = self._determine_query_strategy(query, drug_name)

        self.sources_attempted = list(active_sources)
        self.sources_succeeded = []
        self.sources_failed = []

        coros = []
        for s in active_sources:
            if s == "pubmed":
                coros.append(self._search_pubmed(client, search_query, k))
            elif s == "pubmed_central":
                coros.append(self._search_pubmed_central(client, search_query, k))
            elif s == "openfda":
                coros.append(self._search_openfda(client, drug_name or search_query, query, k))
            elif s == "clinicaltrials":
                coros.append(self._search_clinicaltrials(client, search_query, k))
            elif s == "who":
                coros.append(self._search_who(client, search_query, k))

        try:
            results = await gather_results(coros)
            for s_name, result in zip(active_sources, results):
                if isinstance(result, Exception):
                    logger.error(f"Healthcare source '{s_name}' search error: {result}")
                    self.sources_failed.append(s_name)
                elif isinstance(result, list) and result:
                    passages.extend(result)
                    self.sources_succeeded.append(s_name)
                elif isinstance(result, list) and not result:
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
        stopwords = {
            "is", "a", "an", "the", "for", "and", "or", "of", "in", "to", "on",
            "with", "claim", "this", "that", "these", "those", "cures", "cure",
        }
        words = [w for w in cleaned.split() if len(w) > 1 and w.lower() not in stopwords]
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

            # Fetch abstracts with structured XML parsing
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_res = await client.get(
                fetch_url,
                adapter_name=self.name,
                params={"db": "pubmed", "id": ids_str, "retmode": "xml", "rettype": "abstract"},
            )

            abstracts: Dict[str, str] = {}
            if fetch_res and hasattr(fetch_res, "text") and fetch_res.text:
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(fetch_res.text)
                    for article in root.findall(".//PubmedArticle"):
                        pmid_elem = article.find(".//PMID")
                        if pmid_elem is not None and pmid_elem.text:
                            pmid = pmid_elem.text.strip()
                            abs_texts = article.findall(".//AbstractText")
                            if abs_texts:
                                parts = [elem.text for elem in abs_texts if elem.text]
                                if parts:
                                    abstracts[pmid] = " ".join(parts)
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
                    if abstract_text and len(abstract_text.strip()) > 30:
                        snippet = f"PubMed [PMID:{pm_id}] Study ({pub_date}): {abstract_text}"
                    else:
                        snippet = f"PubMed [PMID:{pm_id}] Title: {title}. Published ({pub_date})."
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
            logger.warning(f"PubMed search failed: {e}")
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

            pmc_abstracts: Dict[str, str] = {}
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
                            text = " ".join(p.text for p in paragraphs if p is not None and p.text)
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
                if abs_text and len(abs_text.strip()) > 30:
                    snippet = f"PMC study [PMC{pmc_uid}] ({pub_date}): {abs_text}"
                else:
                    snippet = f"PMC article [PMC{pmc_uid}] ({pub_date}): {title}."
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
            logger.warning(f"PubMed Central search failed: {e}")
            return []

    async def _search_openfda(
        self, client: ResilientHttpClient, term: str, query: str, k: int
    ) -> List[Passage]:
        try:
            # Clean and prepare search term: use full entity name or clean multi-word query
            clean_term = term.strip() if term and term.strip() else query.strip()
            # Remove punctuation except hyphen
            clean_term = re.sub(r"[^\w\s\-]", " ", clean_term).strip()
            if not clean_term:
                return []

            # Construct flexible FDA query supporting generic, brand, or general search
            fda_query = f'openfda.generic_name:"{clean_term}" OR openfda.brand_name:"{clean_term}" OR openfda.substance_name:"{clean_term}"'

            params: Dict[str, Any] = {"search": fda_query, "limit": k}
            if self.openfda_key:
                params["api_key"] = self.openfda_key

            res = await client.get(
                "https://api.fda.gov/drug/label.json",
                adapter_name=self.name,
                params=params,
            )

            data = []
            try:
                data = res.json().get("results", [])
            except Exception:
                # If structured search failed, try plain query
                params["search"] = clean_term
                res2 = await client.get(
                    "https://api.fda.gov/drug/label.json",
                    adapter_name=self.name,
                    params=params,
                )
                try:
                    data = res2.json().get("results", [])
                except Exception:
                    data = []

            passages = []
            for item in data:
                openfda_block = item.get("openfda", {})
                brand_list = openfda_block.get("brand_name", [])
                generic_list = openfda_block.get("generic_name", [])
                brand = brand_list[0] if brand_list else (generic_list[0] if generic_list else clean_term.title())

                indications = item.get("indications_and_usage", [])
                purpose = item.get("purpose", [])
                desc_list = item.get("description", [])

                content_parts = []
                if indications:
                    content_parts.append(f"Indications: {indications[0]}")
                if purpose:
                    content_parts.append(f"Purpose: {purpose[0]}")
                if desc_list and not content_parts:
                    content_parts.append(desc_list[0])

                desc = " | ".join(content_parts) if content_parts else "Official FDA approved drug label information."
                item_id = item.get("id") or item.get("set_id") or re.sub(r"\W+", "_", brand.lower())

                passages.append(
                    Passage(
                        title=f"FDA Official Drug Label: {brand}",
                        source="openfda",
                        url=f"https://api.fda.gov/drug/label.json?id={item_id}",
                        publication_date="FDA Registered",
                        snippet=f"FDA Drug Label for {brand}: {desc[:400]}",
                        source_id=f"openfda_{item_id}",
                        relevance_score=0.80,
                    )
                )
            return passages
        except Exception as e:
            logger.warning(f"OpenFDA search failed: {e}")
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
                        snippet=f"Clinical Study [{nctId}] (Status: {status}): {title}. Summary: {summary[:350]}",
                        source_id=f"clinicaltrials_{nctId}",
                        relevance_score=0.70,
                    )
                )
            return passages
        except Exception as e:
            logger.warning(f"ClinicalTrials search failed: {e}")
            return []

    async def _search_who(
        self, client: ResilientHttpClient, search_query: str, k: int
    ) -> List[Passage]:
        """Search World Health Organization (WHO) indicators and global health data."""
        try:
            # Query WHO Global Health Observatory API
            res = await client.get(
                "https://ghoapi.azureedge.net/api/Indicator",
                adapter_name=self.name,
                params={"$top": 20},
            )
            data = res.json().get("value", [])
            passages = []
            q_terms = [w.lower() for w in search_query.split() if len(w) > 2]
            for item in data:
                ind_name = item.get("IndicatorName", "")
                ind_code = item.get("IndicatorCode", "")
                ind_lower = ind_name.lower()
                if any(t in ind_lower for t in q_terms):
                    passages.append(
                        Passage(
                            title=f"WHO Health Observatory: {ind_name}",
                            source="who",
                            url=f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{ind_code}",
                            publication_date="WHO Official",
                            snippet=f"WHO Global Health Observatory Indicator [{ind_code}]: {ind_name}. Official World Health Organization global health surveillance data.",
                            source_id=f"who_{ind_code.lower()}",
                            relevance_score=0.75,
                        )
                    )
                if len(passages) >= k:
                    break
            return passages
        except Exception as e:
            logger.warning(f"WHO search failed: {e}")
            return []
